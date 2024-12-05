"""Base module class with PL integration."""

from typing import Optional

import lightning
import torch
from torch import nn

from ... import defaults


class ModuleOutput(nn.Module):
    """Output for forward passes.

    Other than the output tensor itself, all the other fields are nullable and
    can be omitted safely."""

    output: torch.Tensor
    hidden: Optional[torch.Tensor]  # For RNNs.
    cell: Optional[torch.Tensor]  # For LSTMs in particular.
    embedded: Optional[torch.Tensor]

    def __init__(self, output, hidden=None, cell=None, embedded=None):
        super().__init__()
        self.register_buffer("output", output)
        self.register_buffer("hidden", hidden)
        self.register_buffer("cell", cell)
        self.register_buffer("embedded", embedded)

    @property
    def has_hidden(self) -> bool:
        return self.hidden is not None

    @property
    def has_cell(self) -> bool:
        return self.cell is not None

    @property
    def has_embedded(self) -> bool:
        return self.embedded is not None


class BaseModule(lightning.LightningModule):
    # Sizes.
    num_embeddings: int
    # Regularization arguments.
    dropout: float
    # Model arguments.
    embeddings: nn.Embedding
    embedding_size: int
    hidden_size: int
    layers: int
    # Constructed inside __init__.
    dropout_layer: nn.Dropout

    def __init__(
        self,
        *,
        embeddings,
        embedding_size,
        num_embeddings,
        dropout=defaults.DROPOUT,
        layers=defaults.ENCODER_LAYERS,
        hidden_size=defaults.HIDDEN_SIZE,
        **kwargs,  # Ignored.
    ):
        super().__init__()
        self.dropout = dropout
        self.embeddings = embeddings
        self.embedding_size = embedding_size
        self.num_embeddings = num_embeddings
        self.layers = layers
        self.hidden_size = hidden_size
        self.dropout_layer = nn.Dropout(p=self.dropout, inplace=True)

    def embed(self, symbols: torch.Tensor) -> torch.Tensor:
        """Embeds the source symbols and adds positional encodings.

        Args:
            symbols (torch.Tensor): batch of symbols to embed of shape
                B x seq_len.

        Returns:
            torch.Tensor: embedded tensor of shape B x seq_len x embed_dim.
        """
        embedded = self.embeddings(symbols)
        self.dropout_layer(embedded)
        return embedded

    @property
    def output_size(self) -> int:
        raise NotImplementedError
