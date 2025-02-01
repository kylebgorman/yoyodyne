"""RNN module classes."""

import abc
import dataclasses
from typing import Optional

import torch
from torch import nn

from ... import data, defaults, special
from . import attention, base

# Helper classes.


@dataclasses.dataclass
class RNNState(base.BaseState):

    hiddens: Optional[torch.Tensor] = None
    cell: Optional[torch.Tensor] = None


class StateGRU(nn.GRU):
    """Patches GRU API to work with RNNState."""

    def forward(self, state: RNNState) -> RNNState:
        output, hiddens = super().forward(state.tensor, state.hiddens)
        return RNNState(output, hiddens)


class StateLSTM(nn.LSTM):
    """Patches LSTM API to work with RNNState."""

    def forward(self, state: RNNState) -> RNNState:
        # The whole thing has to be null; `(None, None)` won't do.
        if state.hiddens is None and state.cell is None:
            second = None
        else:
            second = (state.hiddens, state.cell)
        output, (hiddens, cell) = super().forward(state.tensor, second)
        return RNNState(output, hiddens, cell)


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
        # Packs embedded source symbols.
        state = RNNState(
            nn.utils.rnn.pack_padded_sequence(
                self.embed(source.padded),
                source.lengths(),
                batch_first=True,
                enforce_sorted=False,
            )
        )
        state = self.module(state)
        # Unpacks the output.
        state.tensor, _ = nn.utils.rnn.pad_packed_sequence(
            state.tensor,
            batch_first=True,
            padding_value=special.PAD_IDX,
        )
        return state

    @property
    def output_size(self) -> int:
        return self.hidden_size * self.num_directions


class GRUEncoder(RNNEncoder):
    """GRU encoder."""

    def get_module(self) -> StateGRU:
        return StateGRU(
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

    def get_module(self) -> StateLSTM:
        return StateLSTM(
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


# These patched modules are only needed for decoder modules.


class RNNDecoder(RNNModule):
    """Abstract base class for RNN decoders.

    This implementation is inattentive; it uses the last (non-padding) hidden
    state of the encoder as the input to the decoder.
    """

    def __init__(self, decoder_input_size, *args, **kwargs):
        self.decoder_input_size = decoder_input_size
        super().__init__(*args, **kwargs)

    def forward(
        self,
        state: RNNState,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> RNNState:
        """Single decode pass.

        Args:
            state (RNNState).
            encoder_out (torch.Tensor): encoded input sequence of shape
                B x seq_len x encoder_dim.
            encoder_mask (torch.Tensor): mask for the encoded input batch of
                shape B x seq_len.

        Returns:
            RNNState.
        """
        embedded = self.embed(state.tensor)
        last_encoder_out = self._last_encoder_out(encoder_out, encoder_mask)
        # Overwrites the state with the concatenation of the embedding
        # and end-of-string encodings.
        state.tensor = torch.cat((embedded, last_encoder_out), dim=2)
        state = self.module(state)
        # Applies dropout to the resulting encoding.
        state.tensor = self.dropout_layer(state.tensor)
        return state

    @staticmethod
    def _last_encoder_out(
        encoder_out: torch.Tensor, encoder_mask: torch.Tensor
    ) -> torch.Tensor:
        """Gets the encoding at the first END for each tensor.

        Args:
            encoder_out (torch.Tensor): encoded input sequence of shape
                B x seq_len x encoder_dim.
            encoder_mask (torch.Tensor): mask for the encoded input batch of
                shape B x seq_len.

        Returns:
            torch.Tensor: indices of shape B x 1 x encoder_dim.
        """
        # Gets the index of the last unmasked tensor.
        # -> B.
        last_idx = (~encoder_mask).sum(dim=1) - 1
        # -> B x 1 x encoder_dim.
        last_idx = last_idx.view(encoder_out.size(0), 1, 1).expand(
            -1, -1, encoder_out.size(2)
        )
        # -> B x 1 x encoder_dim.
        return encoder_out.gather(1, last_idx)

    @property
    def output_size(self) -> int:
        return self.hidden_size


class GRUDecoder(RNNDecoder):
    """GRU decoder."""

    def get_module(self) -> StateGRU:
        return StateGRU(
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

    def get_module(self) -> StateLSTM:
        return StateLSTM(
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

    Subsequent concrete implementations use the attention module to
    differentially attend to different parts of the encoder output.
    """

    def __init__(self, attention_input_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention = attention.Attention(
            attention_input_size, self.hidden_size
        )

    def forward(
        self,
        state: RNNState,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> RNNState:
        embedded = self.embed(state.tensor)
        context, _ = self.attention(
            state.hiddens.transpose(0, 1), encoder_out, encoder_mask
        )
        # Ovewrites the state with the concatenation of the embedding and
        # the context.
        state.tensor = torch.cat((embedded, context), dim=2)
        state = self.module(state)
        # Applies dropout to the resulting encoding.
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
