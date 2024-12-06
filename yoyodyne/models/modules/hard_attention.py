"""Hard attention module classes."""

from typing import Optional

import torch
from torch import nn

from ... import defaults
from . import base, rnn


class HardAttentionModuleOutput(nn.Module):
    """Output for forward passes.

    This is similar to `base.ModuleOutput` but has different field names."""

    emissions: torch.Tensor
    transitions: torch.Tensor
    hidden: torch.Tensor  # Here mandatory because this works with RNNs.
    cell: Optional[torch.Tensor]  # For LSTMs in particular.

    def __init__(self, emissions, transitions, hidden, cell=None):
        super().__init__()
        self.register_buffer("emissions", emissions)
        self.register_buffer("transitions", transitions)
        self.register_buffer("hidden", hidden)
        self.register_buffer("cell", cell)


class HardAttentionRNNDecoder(rnn.RNNDecoder):
    """Base module for zeroth-order HMM hard attention RNN decoders."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Activates emission probs.
        self.output_proj = nn.Sequential(
            nn.Linear(self.output_size, self.output_size), nn.Tanh()
        )
        # Projects transition probabilities to depth of module.
        self.scale_encoded = nn.Linear(
            self.decoder_input_size, self.hidden_size
        )

    def _emissions(
        self, decoded: torch.Tensor, encoder_output
    ) -> torch.Tensor:
        """Computes emission probabilities over each hidden state.

        Args:
            decoded (torch.Tensor): previously output symbol of shape B x 1.
            encoder_output (torch.Tensor): encoded input sequence of shape
                B x seq_len x encoder_dim.
        """
        output = decoded.expand(-1, encoder_output.shape[1], -1)
        output = torch.cat((output, encoder_output), dim=-1)
        output = self.output_proj(output)
        return output

    def _transitions(
        self,
        decoded: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Creates alignment matrix for current timestep.

        Given the current encoder repreesentation and the decoder
        representation at the current time step, this calculates the alignment
        scores between all potential source sequence pairings. These
        alignments are used to predict the likelihood of state transitions
        for the output.

        After:
            Wu, S. and Cotterell, R. 2019. Exact hard monotonic attention for
            character-level transduction. In _Proceedings of the 57th Annual
            Meeting of the Association for Computational Linguistics_, pages
            1530-1537.

        Args
            decoded (torch.Tensor): output from decoder for current timesstep
                of shape B x 1 x decoder_dim.
            encoder_output (torch.Tensor): encoded input sequence of shape
                B x seq_len x encoder_dim.
            encoder_mask (torch.Tensor): mask for the encoded input batch of
                shape B x seq_len.

        Returns:
            torch.Tensor: alignment scores across the source sequence of shape
                B x seq_len.
        """
        alignment_scores = torch.bmm(
            self.scale_encoded(encoder_output), decoded.transpose(1, 2)
        ).squeeze(-1)
        # Gets probability of alignments.
        alignment_probs = nn.functional.softmax(alignment_scores, dim=-1)
        # Mask padding.
        alignment_probs = alignment_probs * (~encoder_mask) + defaults.EPSILON
        alignment_probs = alignment_probs / alignment_probs.sum(
            dim=-1, keepdim=True
        )
        # Expands over all time steps. Log probs for quicker computation.
        return (
            alignment_probs.log()
            .unsqueeze(1)
            .expand(-1, encoder_output.shape[1], -1)
        )

    @property
    def output_size(self) -> int:
        return self.decoder_input_size + self.hidden_size


class HardAttentionGRUDecoder(HardAttentionRNNDecoder):
    """Zeroth-order HMM hard attention GRU decoder."""

    def get_module(self) -> nn.GRU:
        return nn.GRU(
            self.embedding_size,
            self.hidden_size,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
            num_layers=self.layers,
        )

    def forward(
        self,
        decoder_input: base.ModuleOutput,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> base.ModuleOutput:
        """Single decode pass.

        Args:
            decoder_input (base.ModuleOutput): previously decoded symbol
                of shape B x 1, last hidden state from the decoder of shape
                1 x B x decoder_dim, and the cell state of the same dimensions
                as the hidden state.
            encoder_output (torch.Tensor): encoded input sequence of shape
                (B x seq_len x encoder_dim).
            encoder_mask (torch.Tensor): mask for the encoded input batch of
                shape (B x seq_len).

        Returns:
            base.ModuleOutput: step-wise emission probabilities, alignment
                matrix, and hidden states of decoder.
        """
        embedded = self.embed(decoder_input.output)
        output, hidden = self.module(embedded, decoder_input.hidden)
        emissions = self._emissions(output, encoder_output)
        transitions = self._transitions(output, encoder_output, encoder_mask)
        return HardAttentionModuleOutput(emissions, transitions, hidden)

    @property
    def name(self) -> str:
        return "hard attention GRU"


class HardAttentionLSTMDecoder(HardAttentionRNNDecoder):
    """Zeroth-order HMM hard attention LSTM decoder."""

    def get_module(self) -> nn.LSTM:
        return nn.LSTM(
            self.embedding_size,
            self.hidden_size,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
            num_layers=self.layers,
        )

    # Implementationally this is identical to the GRU except the presence of
    # the cell state.

    def forward(
        self,
        decoder_input: base.ModuleOutput,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> base.ModuleOutput:
        """Single decode pass.

        Args:
            decoder_input (base.ModuleOutput): previously decoded symbol
                of shape B x 1, last hidden state from the decoder of shape
                1 x B x decoder_dim, and the cell state of the same dimensions
                as the hidden state.
            encoder_output (torch.Tensor): encoded input sequence of shape
                (B x seq_len x encoder_dim).
            encoder_mask (torch.Tensor): mask for the encoded input batch of
                shape (B x seq_len).

        Returns:
            base.ModuleOutput: step-wise emission probabilities, alignment
                matrix, and hidden states of decoder.
        """
        embedded = self.embed(decoder_input.output)
        output, (hidden, cell) = self.module(
            embedded, (decoder_input.hidden, decoder_input.cell)
        )
        emissions = self._emissions(output, encoder_output)
        transitions = self._transitions(output, encoder_output, encoder_mask)
        return HardAttentionModuleOutput(emissions, transitions, hidden, cell)

    @property
    def name(self) -> str:
        return "hard attention LSTM"


class ContextHardAttentionRNNDecoder(HardAttentionRNNDecoder):
    """Base module for first-order HMM hard attention RNN decoder."""

    def __init__(self, attention_context, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = attention_context
        # Window size must include center and both sides.
        self.alignment_proj = nn.Linear(
            self.hidden_size * 2, (attention_context * 2) + 1
        )

    def _transitions(
        self,
        decoded: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Creates alignment matrix for current timestep.

        Args
            decoded (torch.Tensor): output from decoder for current timesstep
                of shape B x 1 x decoder_dim.
            encoder_output (torch.Tensor): encoded input sequence of shape
                B x seq_len x encoder_dim.
            encoder_mask (torch.Tensor): mask for the encoded input batch of
                shape B x seq_len.

        Returns:
            torch.Tensor: alignment scores across the source sequence of shape
                B x seq_len.
        """
        # Matrix multiplies encoding and decoding for alignment
        # representations. See: https://aclanthology.org/P19-1148/.
        # Expands decoded to concatenate with alignments.
        decoded = decoded.expand(-1, encoder_output.shape[1], -1)
        # -> B x seq_len.
        alignment_scores = torch.cat(
            (self.scale_encoded(encoder_output), decoded), dim=2
        )
        alignment_scores = self.alignment_proj(alignment_scores)
        alignment_probs = nn.functional.softmax(alignment_scores, dim=-1)
        # Limits context to window of self.delta (context length).
        alignment_probs = alignment_probs.split(1, dim=1)
        alignment_probs = torch.cat(
            [
                nn.functional.pad(
                    t,
                    (
                        -self.delta + i,
                        encoder_mask.shape[1] - (self.delta + 1) - i,
                    ),
                )
                for i, t in enumerate(alignment_probs)
            ],
            dim=1,
        )
        # Gets probability of alignments, masking padding.
        alignment_probs = (
            alignment_probs * (~encoder_mask).unsqueeze(1) + defaults.EPSILON
        )
        alignment_probs = alignment_probs / alignment_probs.sum(
            dim=-1, keepdim=True
        )
        # Log probs for quicker computation.
        return alignment_probs.log()


class ContextHardAttentionGRUDecoder(
    ContextHardAttentionRNNDecoder, HardAttentionGRUDecoder
):
    """First-order HMM hard attention GRU decoder."""

    @property
    def name(self) -> str:
        return "contextual hard attention GRU"


class ContextHardAttentionLSTMDecoder(
    ContextHardAttentionRNNDecoder, HardAttentionLSTMDecoder
):
    """First-order HMM hard attention LSTM decoder."""

    @property
    def name(self) -> str:
        return "contextual hard attention LSTM"
