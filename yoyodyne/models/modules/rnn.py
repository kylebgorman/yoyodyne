"""RNN module classes.

RNNModule is the base class, and records bidirectionality.

We abstract away from the different formats (for inputs and outputs) used by
GRUs and LSTMs; the latter also tracks "cell state" and stores it as part of
a tuple of tensors along with the hidden state. RNNState is used to represent
the state of an RNN, hiding this detail. For encoder modules,
GRUEncoderModule and LSTMEncoderModule wrap nn.GRU and nn.LSTM, respectively,
and taking responsibility for padding the packed sequence; GRUDecoderModule
and LSTMDecoderModule are similar wrappers for decoder modules.
"""

import abc
import dataclasses
from typing import Optional

import torch
from torch import nn

from ... import data, defaults, special
from . import attention, base


class RNNModule(base.BaseModule):
    """Abstract base class for RNN modules.

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

    @abc.abstractmethod
    def get_module(self) -> nn.RNNBase: ...

    @property
    def num_directions(self) -> int:
        return 2 if self.bidirectional else 1


@dataclasses.dataclass
class RNNState(nn.Module):
    """Represents the state of an RNN."""

    sequence: torch.Tensor
    hiddens: torch.Tensor
    # For LSTMs only.
    cell: Optional[torch.Tensor] = None

    # FIXME debugging only
    def __post_init__(self):
        print("sequence:", self.sequence.shape)
        print("hiddens:", self.hiddens.shape)
        if self.cell is not None:
            print("cell:", sell.cell.shape)


class RNNEncoderModule:
    """Patches RNN encoder modules to deal with padding packed sequences.

    In contrast to the decoder, whose initial state is represented by learned
    parameters, the initial state of the encoder is effectively zero because
    no hidden state (or cell state in the case of LSTMs) is provided.
    """

    @staticmethod
    def _pad_packed(sequence: nn.utils.rnn.PackedSequence) -> torch.Tensor:
        packed, _ = nn.utils.rnn.pad_packed_sequence(
            sequence,
            batch_first=True,
            padding_value=special.PAD_IDX,
        )
        return packed

    @abc.abstractmethod
    def forward(self, sequence: nn.utils.rnn.PackedSequence) -> RNNState: ...


class GRUEncoderModule(nn.GRU, RNNEncoderModule):

    def forward(self, sequence: nn.utils.rnn.PackedSequence) -> RNNState:
        packed, hiddens = super().forward(sequence)
        print("Encoder:")
        return RNNState(self._pad_packed(packed), hiddens)


class LSTMEncoderModule(nn.LSTM, RNNEncoderModule):

    def forward(self, sequence: nn.utils.rnn.PackedSequence) -> RNNState:
        print("Encoder:")
        packed, (hiddens, cell) = super().forward(sequence)
        return RNNState(self._pad_packed(packed), hiddens, cell)


class RNNEncoder(RNNModule):
    """Abstract base class for RNN encoders."""

    def forward(self, source: data.PaddedTensor) -> RNNState:
        """Encodes the input.

        Args:
            source (data.PaddedTensor): source padded tensors and mask
                for source, of shape B x seq_len x 1.

        Returns:
            RNNState.
        """
        embedded = self.embed(source.padded)
        sequence = self._pack_padded(embedded, source.lengths())
        return self.module(sequence)

    @staticmethod
    def _pack_padded(
        sequence: torch.Tensor, lengths: torch.Tensor
    ) -> nn.utils.rnn.PackedSequence:
        return nn.utils.rnn.pack_padded_sequence(
            sequence,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )

    @property
    def output_size(self) -> int:
        return self.hidden_size * self.num_directions


class GRUEncoder(RNNEncoder):
    """GRU encoder."""

    def get_module(self) -> GRUEncoderModule:
        return GRUEncoderModule(
            self.embedding_size,
            self.hidden_size,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
            num_layers=self.layers,
        )

    @property
    def name(self) -> str:
        return "GRU"


class LSTMEncoder(RNNEncoder):
    """LSTM encoder."""

    def get_module(self) -> LSTMEncoderModule:
        return LSTMEncoderModule(
            self.embedding_size,
            self.hidden_size,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
            num_layers=self.layers,
        )

    @property
    def name(self) -> str:
        return "LSTM"


class GRUDecoderModule(nn.GRU):
    """Patches GRU API to work with RNNState."""

    def forward(self, state: RNNState) -> RNNState:
        output, hiddens = super().forward(state.sequence, state.hiddens)
        print("Decoder:")
        return RNNState(output, hiddens)


class LSTMDecoderModule(nn.LSTM):
    """Patches LSTM API to work with RNNState."""

    def forward(self, state: RNNState) -> RNNState:
        output, (hiddens, cell) = super().forward(
            state.sequence, (state.hiddens, state.cell)
        )
        print("Decoder:")
        return RNNState(output, hiddens, cell)


class RNNDecoder(RNNModule):
    """Abstract base class for RNN decoders.

    This implementation is inattentive; it uses the last (non-padding) hidden
    state of the encoder as the input to the decoder."""

    def __init__(self, decoder_input_size, *args, **kwargs):
        self.decoder_input_size = decoder_input_size
        super().__init__(*args, **kwargs)

    def forward(
        self,
        state: RNNState,
        encoded: torch.Tensor,
        mask: torch.Tensor,
    ) -> RNNState:
        """Single decode pass.

        Args:
            state (RNNState).
            encoded (torch.Tensor): encoded input sequence of shape
                B x seq_len x encoder_dim.
            mask (torch.Tensor): mask for the encoded input batch of
                shape B x seq_len.

        Returns:
            RNNState.
        """
        embedded = self.embed(state.tensor)
        last_encoded = self._last_encoded(encoded, mask)
        # Overwrites the state with the concatenation of the embedding
        # and end-of-string encodings.
        state.tensor = torch.cat((embedded, last_encoder), dim=2)
        state = self.module(state)
        state.tensor = self.dropout_layer(state.tensor)
        return state

    @staticmethod
    def _last_encoded(
        encoded: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Gets the encoding at the first END for each tensor.

        Args:
            encoded (torch.Tensor): encoded input sequence of shape
                B x seq_len x encoder_dim.
            mask (torch.Tensor): mask for the encoded input batch of
                shape B x seq_len.

        Returns:
            torch.Tensor: indices of shape B x 1 x encoder_dim.
        """
        # Gets the index of the last unmasked tensor.
        # -> B.
        last_idx = (~mask).sum(dim=1) - 1
        # -> B x 1 x encoder_dim.
        last_idx = last_idx.view(encoded.size(0), 1, 1).expand(
            -1, -1, encoded.size(2)
        )
        # -> B x 1 x encoder_dim.
        return encoded.gather(1, last_idx)

    @property
    def output_size(self) -> int:
        return self.hidden_size


class GRUDecoder(RNNDecoder):
    """GRU decoder."""

    def get_module(self) -> GRUDecoderModule:
        return GRUDecoderModule(
            self.decoder_input_size + self.embedding_size,
            self.hidden_size,
            num_layers=self.layers,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

    @property
    def name(self) -> str:
        return "GRU"


class LSTMDecoder(RNNDecoder):
    """LSTM decoder."""

    def get_module(self) -> LSTMDecoderModule:
        return LSTMDecoderModule(
            self.decoder_input_size + self.embedding_size,
            self.hidden_size,
            num_layers=self.layers,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

    @property
    def name(self) -> str:
        return "LSTM"


class AttentiveRNNDecoder(RNNDecoder):
    """Abstract base class for attentive RNN decoders.

    The attention module differentially attends to different parts of the
    encoder output.
    """

    def __init__(self, attention_input_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention = attention.Attention(
            attention_input_size, self.hidden_size
        )

    def forward(
        self,
        state: RNNState,
        encoded: torch.Tensor,
        mask: torch.Tensor,
    ) -> RNNState:
        embedded = self.embed(state.sequence)
        context, _ = self.attention(
            state.hiddens.transpose(0, 1), encoded, mask
        )
        # Ovewrites the state with the concatenation of the embedding and
        # the context.
        state.tensor = torch.cat((embedded, context), dim=2)
        state = self.module(state)
        state.tensor = self.dropout_layer(state.tensor)
        return state


class AttentiveGRUDecoder(AttentiveRNNDecoder, GRUDecoder):
    """Attentive GRU decoder."""

    @property
    def name(self) -> str:
        return "attentive GRU"


class AttentiveLSTMDecoder(AttentiveRNNDecoder, LSTMDecoder):
    """Attentive LSTM decoder."""

    @property
    def name(self) -> str:
        return "attentive LSTM"
