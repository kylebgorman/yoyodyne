"""Beam search classes.

A BeamCell is a (possibly partial) hypothesis containing the decoder output,
the symbol sequence, and the hypothesis's log-likelihood. BeamCells can
generate their candidate extensions (in the form of new BeamCells) when
provided with additional decoder output; they also know when they have reached
a final state (i.e., when END has been generated).

A Beam holds a collection of BeamCells; it knows when all these hypotheses have
reached a final state.

A BeamHelper is a heap of possible extensions; heap size is constrained by
beam width. "Recording" a cell and the associated decoder output causes its
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
* Not much attention has been paid to keeping data on device.

Sample usage:

    def beam_decode(
        self, encoder_output: torch.Tensor, encoder_mask: torch.Tensor
    ) -> torch.Tensor:
        beam = Beam([self._initial_input(self.batch_size)]))))
        for t in range(max_length):
            with BeamHelper(self.beam_width) as helper:
                for cell in beam.cells:
                    decoder_input = cell.decoder_output.to(self.device)
                    decoder_output = self.decode_one(decoder_input)
                    helper.record(cell, decoder_outout)
            beam = helper.update()
            if beam.final:
                break
        predictions = nn.utils.rnn.pad_sequence(
            [torch.tensor(cell.symbols) for cell in beam.cells],
            padding_value=special.PAD_IDX,
            device=self.device,
        ).unsqueeze(0)
        loglikes = torch.tensor(
            [cell.loglike) for cell in beam.cells],
            device=self.device,
        )
        return predictions, loglikes
"""

from __future__ import annotations

import dataclasses
import heapq

from typing import Iterator, List

from torch import nn

from . import modules
from .. import special


@dataclasses.dataclass(order=True)
class BeamCell:
    """Represents a (partial) hypotheses in the beam search.

    The class is sorted by the log-likelihood field.

    Args:
        decoder_input (modules.ModuleOutput).
    """

    decoder_input: modules.ModuleOutput = dataclasses.field(compare=False)
    symbols: List[int] = dataclasses.field(
        compare=False, default_factory=lambda: [special.IDX]
    )
    loglike: float = dataclasses.field(default_factory=float)  # AKA 0.0.

    def extensions(
        self, decoder_output: modules.ModuleOutput
    ) -> Iterator[BeamCell]:
        if self.final:
            # No extensions are possible.
            yield self
            return
        logits = decoder_output.output.squeeze((0, 1))
        predictions = nn.functional.log_softmax(logits, dim=0).cpu()
        for last_symbol, log_prob in enumerate(predictions):
            yield BeamCell(
                decoder_output,
                self.symbols + [last_symbol],
                self.loglike + log_prob,
            )

    @property
    def final(self) -> bool:
        return self.symbols[-1] == special.END_IDX


class Beam:
    """Represents the current state of the beam."""

    cells: List[BeamCell]

    def __init__(self, cells: List[BeamCell], beam_width: int):
        self.cells = cells

    def __len__(self) -> int:
        return len(self.cells)

    @property
    def final(self) -> bool:
        return all(cell.final for cell in self.cells)


class BeamHelper:
    """Helps extend a beam."""

    beam_width: int
    heap: List[BeamCell] = []

    def __init__(self, beam_width: int):
        self.beam_width = beam_width
        self.heap = []

    def __enter__(self) -> BeamHelper:
        self.heap.clear()
        return self

    def record(
        self, cell: BeamCell, decoder_output: modules.ModuleOutput
    ) -> None:
        for new_cell in cell.extensions(decoder_output):
            if len(self.new_beam) < self.beam_width:
                heapq.heappush(self.heap, new_cell)
            else:
                heapq.heappushpop(self.heap, new_cell)

    def __exit__(self):
        # Sorts hypotheses so the minimum log-likelihood is the first
        # element; we think this is faster than calling heapq.nlargest.
        self.heap.sort(reverse=True)

    def update(self) -> Beam:
        return Beam(self.heap)
