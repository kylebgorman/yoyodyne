"""RNN module classes."""

import torch
from torch import nn

from ... import data, defaults, special
from . import attention, base


class RNNModule(base.BaseModule):
    """Base class for RNN modules.

    Args:
        bidirectional (bool).
        *args: passed to superclass.
        **kwargs: passed to superclass.
    """

    bidirectional: bool

    def __init__(
        self,
        *args,
        bidirectional=defaults.BIDIRECTIONAL,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bidirectional = bidirectional
        self.module = self.get_module()

    @property
    def num_directions(self) -> int:
        return 2 if self.bidirectional else 1

    def get_module(self):
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError


class RNNEncoder(RNNModule):
    """Base class for RNN encoders."""

    @property
    def output_size(self) -> int:
        return self.hidden_size * self.num_directions


class GRUEncoder(RNNEncoder):
    """GRU encoder."""

    def get_module(self) -> nn.GRU:
        return nn.GRU(
            self.embedding_size,
            self.hidden_size,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
            num_layers=self.layers,
        )

    def forward(self, source: data.PaddedTensor) -> base.ModuleOutput:
        """Encodes the input.

        Args:
            source (data.PaddedTensor): source padded tensors and mask
                for source, of shape B x seq_len x 1.

        Returns:
            base.ModuleOutput.
        """
        embedded = self.embed(source.padded)
        # Packs embedded source symbol into a PackedSequence.
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            source.lengths(),
            batch_first=True,
            enforce_sorted=False,
        )
        # -> B x seq_len x encoder_dim, hidden
        packed_outs, hidden = self.module(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outs,
            batch_first=True,
            padding_value=special.PAD_IDX,
            total_length=None,
        )
        return base.ModuleOutput(encoded, hidden)

    @property
    def name(self) -> str:
        return "GRU"


class LSTMEncoder(RNNEncoder):
    """LSTM encoder."""

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

    def forward(self, source: data.PaddedTensor) -> base.ModuleOutput:
        """Encodes the input.

        Args:
            source (data.PaddedTensor): source padded tensors and mask
                for source, of shape B x seq_len x 1.

        Returns:
            base.ModuleOutput.
        """
        embedded = self.embed(source.padded)
        # Packs embedded source symbol into a PackedSequence.
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            source.lengths(),
            batch_first=True,
            enforce_sorted=False,
        )
        # -> B x seq_len x encoder_dim, hidden
        packed_outs, (hidden, cell) = self.module(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outs,
            batch_first=True,
            padding_value=special.PAD_IDX,
            total_length=None,
        )
        return base.ModuleOutput(encoded, hidden, cell)

    @property
    def name(self) -> str:
        return "LSTM"


class RNNDecoder(RNNModule):
    """Base class for RNN decoders."""

    def __init__(self, decoder_input_size, *args, **kwargs):
        self.decoder_input_size = decoder_input_size
        super().__init__(*args, **kwargs)

    @staticmethod
    def _last_encoder_output(
        encoder_output: torch.Tensor, encoder_mask: torch.Tensor
    ) -> torch.Tensor:
        """Computes encoder output at the last non-masked symbol.

        Args:
            encoder_output (torch.Tensor): encoded input sequence of shape
                B x seq_len x encoder_dim.
            encoder_mask (torch.Tensor): mask for the encoded input batch of
                shape B x seq_len.

        Returns:
            torch.Tensor.
        """
        # Get the index of the last unmasked tensor.
        # -> B.
        last_encoder_output_idxs = (~encoder_mask).sum(dim=1) - 1
        # -> B x 1 x 1.
        last_encoder_output_idxs = last_encoder_output_idxs.view(
            encoder_output.size(0), 1, 1
        )
        # -> 1 x 1 x encoder_dim. This indexes the last non-padded dimension.
        last_encoder_output_idxs = last_encoder_output_idxs.expand(
            -1, -1, encoder_output.size(-1)
        )
        # -> B x 1 x encoder_dim.
        return torch.gather(encoder_output, 1, last_encoder_output_idxs)

    @property
    def output_size(self) -> int:
        return self.hidden_size


class GRUDecoder(RNNDecoder):
    """GRU decoder."""

    def get_module(self) -> nn.GRU:
        return nn.GRU(
            self.decoder_input_size + self.embedding_size,
            self.hidden_size,
            num_layers=self.layers,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=self.bidirectional,
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
                1 x B x decoder_dim.
            encoder_output (torch.Tensor): encoded input sequence of shape
                B x seq_len x encoder_dim.
            encoder_mask (torch.Tensor): mask for the encoded input batch of
                shape B x seq_len.

        Returns:
            base.ModuleOutput: decoder output, and the previous hidden states
                from the decoder RNN.
        """
        embedded = self.embed(decoder_input.output)
        last_encoder_output = self._last_encoder_output(
            encoder_output, encoder_mask
        )

        output, hidden = self.module(
            torch.cat((embedded, last_encoder_output), 2),
            (decoder_input.hidden, decoder_input.cell),
        )
        self.dropout_layer(output)
        return base.ModuleOutput(output, hidden)

    @property
    def name(self) -> str:
        return "GRU"


class LSTMDecoder(RNNDecoder):
    """LSTM decoder."""

    def get_module(self) -> nn.LSTM:
        return nn.LSTM(
            self.decoder_input_size + self.embedding_size,
            self.hidden_size,
            num_layers=self.layers,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=self.bidirectional,
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
                B x seq_len x encoder_dim.
            encoder_mask (torch.Tensor): mask for the encoded input batch of
                shape B x seq_len.

        Returns:
            base.ModuleOutput: decoder output, and the previous hidden state
                and cell state from the decoder RNN.
        """
        embedded = self.embed(decoder_input.output)
        last_encoder_output = self._last_encoder_output(
            encoder_output, encoder_mask
        )
        output, (hidden, cell) = self.module(
            torch.cat((embedded, last_encoder_output), 2),
            (decoder_input.hidden, decoder_input.cell),
        )
        self.dropout_layer(output)
        return base.ModuleOutput(output, hidden, cell)

    @property
    def name(self) -> str:
        return "LSTM"


class AttentiveRNNDecoder(RNNDecoder):
    """Base class for attentive RNN decoders."""

    def __init__(self, attention_input_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention = attention.Attention(
            attention_input_size, self.hidden_size
        )


class AttentiveGRUDecoder(AttentiveRNNDecoder, GRUDecoder):
    """Attentive GRU decoder."""

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
                1 x B x decoder_dim.
                the same dimensions as the hidden state.
            encoder_output (torch.Tensor): encoded input sequence of shape
                B x seq_len x encoder_dim.
            encoder_mask (torch.Tensor): mask for the encoded input batch of
                shape B x seq_len.

        Returns:
            base.ModuleOutput: decoder output, and the previous hidden state
                from the decoder RNN.
        """
        embedded = self.embed(decoder_input.output)
        context, _ = self.attention(
            decoder_input.hidden.transpose(0, 1), encoder_output, encoder_mask
        )
        output, hidden = self.module(
            torch.cat((embedded, context), 2), decoder_input.hidden
        )
        self.dropout_layer(output)
        return base.ModuleOutput(output, hidden)

    @property
    def name(self) -> str:
        return "attentive GRU"


class AttentiveLSTMDecoder(AttentiveRNNDecoder, LSTMDecoder):
    """Attentive LSTM decoder."""

    @property
    def name(self) -> str:
        return "attentive LSTM"

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
                B x seq_len x encoder_dim.
            encoder_mask (torch.Tensor): mask for the encoded input batch of
                shape B x seq_len.

        Returns:
            base.ModuleOutput: decoder output, and the previous hidden state
                from the decoder RNN.
        """
        embedded = self.embed(decoder_input.output)
        context, _ = self.attention(
            decoder_input.hidden.transpose(0, 1), encoder_output, encoder_mask
        )
        output, (hidden, cell) = self.module(
            torch.cat((embedded, context), 2),
            (decoder_input.hidden, decoder_input.cell),
        )
        self.dropout_layer(output)
        return base.ModuleOutput(output, hidden, cell)


class HardAttentionRNNDecoder(RNNDecoder):
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
                1 x B x decoder_dim, and (where appropriate) the cell state
                of the same dimensions as the hidden state.
            encoder_output (torch.Tensor): encoded input sequence of shape
                (B x seq_len x encoder_dim).
            encoder_mask (torch.Tensor): mask for the encoded input batch of
                shape (B x seq_len).

        Returns:
            base.ModuleOutput: step-wise emission probabilities, alignment
                matrix, and hidden states of decoder.
        """
        embedded = self.embed(decoder_input.output)
        decoded, *rest = self.module(embedded, decoder_input.hidden)
        emissions = self._emissions(decoder_input, encoder_output)
        alignments = self._alignments(decoded, encoder_output, encoder_mask)
        return base.ModuleOutput(emissions, *rest, embedded=alignments)

    def _emissions(
        self, output: base.ModuleOutput, encoder_output
    ) -> torch.Tensor:
        """Computes emission probabilities over each hidden state.

        Args:
            output (torch.Tensor): previously output symbol of shape B x 1.
            encoder_output (torch.Tensor): encoded input sequence of shape
                (B x seq_len x encoder_dim).
        """
        output = output.expand(-1, encoder_output.shape[1], -1)
        output = torch.cat([output, encoder_output], dim=-1)
        output = self.output_proj(output)
        return output

    def _alignments(
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
        alignment_probs = alignment_probs * (~encoder_mask) + special.EPSILON
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

    def _alignments(
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
            [self.scale_encoded(encoder_output), decoded], dim=2
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
