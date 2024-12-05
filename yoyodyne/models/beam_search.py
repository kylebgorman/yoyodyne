"""Beam search classes.

A Node is a (possibly partial) hypothesis containing the decoder output,
the symbol sequence, and the hypothesis's log-likelihood. Nodes can
generate their candidate extensions (in the form of new Nodes) when
provided with additional decoder output; they also know when they have reached
a final state (i.e., when END has been generated).

A Beam holds a collection of Nodes; it knows when all these hypotheses have
reached a final state.

A BeamHelper is a heap of possible extensions; heap size is constrained by
beam width. "Recording" a node and the associated decoder output causes its
extensions to be inserted into the heap. It has a context manager interface;
entering the context empties the heap and exiting it "flattens" the heap into
an ordinary list. It can then be used to update the beam.

Current limitations:

* Beam search uses Python's heap implementation; this is reasonably performant
  in cPython (it uses a C extension module where available) but there may be a
  better pure PyTorch solution.
* Beam search assumes a batch size of 1; it is not clear how to extend it to
  larger batches.
* We hard-code the use of log-likelihoods; the addition of two log
  probabilities is equivalent to multiplying real numbers.
* Beam search is designed to support RNN and attentive RNN models; it has not
  yet been generalized for other model types.
* Not much attention has been paid to keeping data on device within this
  module.

For sample usage see `rnn.py`.
"""

from __future__ import annotations

import dataclasses
import heapq

from typing import Iterator, List, Optional

import torch
from torch import nn

from . import modules
from .. import special


# Histories represents the actual beam state. Entries are of the form:
#
# likelihood, [sequence], hidden state
#
# To make the next heap, we first generate a list called `likelihoods`.
# Entries are the forms:
#
# logits, likelihood, [sequence], hidden state


@dataclasses.dataclass(order=True)
class Node:
    """Represents a (partial) hypotheses in the beam search.

    Computations are performed in log-space, so:

    * `+` is equivalent to multiplication in the real domain, and
    * sorting nodes by ascending order of their log-likelihood gives
      hypotheses in descending order of likelihood.
    """

    # This metadata provides default values and ensures that instances are
    # sorted using the `loglike` field.

    loglike: float = 0.0
    symbols: List[int] = dataclasses.field(
        compare=False, default_factory=lambda: [special.START_IDX]
    )
    # TODO: This is written generically to support whatever tensors one might
    # find in the module output of a decoder step. Currently that includes the
    # hidden state for RNNs and the cell state for LSTMs in particular, but it
    # would be straightforward to extend this to support models that also pass
    # embeddings. These fields are all nullable.
    hidden: Optional[torch.Tensor] = dataclasses.field(
        compare=False, default_factory=lambda: None
    )
    cell: Optional[torch.Tensor] = dataclasses.field(
        compare=False, default_factory=lambda: None
    )

    def extensions(
        self, decoder_output: modules.ModuleOutput
    ) -> Iterator[Node]:
        """Generates all extensions of the current node.

        Using the provided decoder output, this generates all nodes which
        extend the current hypothesis by a single symbol.

        Args:
            decoder_output (modules.ModuleOutput): logits, and optional hidden
                and cell state.

        Yields:
            Node: new nodes.
        """
        if self.is_final:
            # No extensions are possible.
            yield self
            return
        # FIXME: is moving this to CPU wise or necessary?
        predictions = nn.functional.log_softmax(decoder_output.output, dim=0)
        for symbol, log_prob in enumerate(predictions):
            yield Node(
                self.loglike + log_prob.item(),
                self.symbols + [symbol],
                decoder_output.hidden,
                decoder_output.cell,
            )

    def decoder_input(self) -> modules.ModuleOutput:
        """Generates input for the decoder.

        Using the current final symbol in the hypothesis, this generates
        the input the decoder will use for the next step.

        The caller will likely want to move the returned value to the current
        device.

        Returns:
            modules.ModuleOutput.
        """
        symbol = torch.tensor([self.symbols[-1]]).unsqueeze(1)
        return modules.ModuleOutput(symbol, self.hidden, self.cell)

    @property
    def is_final(self) -> bool:
        return self.symbols[-1] == special.END_IDX


@dataclasses.dataclass
class Beam:
    """Represents the current state of the beam."""

    nodes: List[Node]

    @classmethod
    def from_initial_input(cls, decoder_output: modules.ModuleOutput) -> Beam:
        """Constructs an initial beam using the decoder output."""
        node = Node(hidden=decoder_output.hidden, cell=decoder_output.cell)
        return cls([node])

    @property
    def is_final(self) -> bool:
        return all(node.is_final for node in self.nodes)


@dataclasses.dataclass
class BeamHelper:
    """Helps extend a beam.

    This helper wraps a min-heap (of a size determined by the beam width) which
    contains candidate nodes. It is intended to be used as a context manager:

    * "entering" the helper clears any previous data.
    * "exiting" it linearizes the heap.

    One can then retrieve a new beam by calling the `beam` method.
    """

    beam_width: int
    heap: List[Node] = dataclasses.field(default_factory=list)

    def __enter__(self) -> BeamHelper:
        self.heap.clear()
        return self

    def record(self, node: Node, decoder_output: modules.ModuleOutput) -> None:
        """Inserts new hypotheses into the heap.

        Using a node of length n and the decoder output, this generates all
        possible nodes of length n + 1 and inserts them into the heap while
        keeping it from growing larger than the beam width specified when it
        was constructed.

        Args:
            node (Node).
            decoder_output (modules.ModuleOutput).
        """
        for new_node in node.extensions(decoder_output):
            if len(self) < self.beam_width:
                heapq.heappush(self.heap, new_node)
            else:
                heapq.heappushpop(self.heap, new_node)

    def __len__(self) -> int:
        return len(self.heap)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        # Sorts hypotheses so the minimum log-likelihood is the first
        # element; we think this is faster than repeatedly calling
        # heapq.nlargest.
        self.heap.sort(reverse=True)

    def beam(self) -> Beam:
        """Constructs a new beam from the heap.

        Returns:
            Beam.
        """
        return Beam(self.heap)
