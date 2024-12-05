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
