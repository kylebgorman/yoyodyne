"""Positional encoding modules for transformers."""

import abc
import math

import lightning
import torch
from torch import nn

from ... import defaults, special


class Error(Exception):
    pass


class BasePositionalEncoding(abc.ABC, lightning.LightningModule):
    """Abstract base class for positional encodings."""

    @abc.abstractmethod
    def forward(
        self,
        embedded: torch.Tensor,
        symbol: torch.Tensor,
    ) -> torch.Tensor: ...

    @property
    @abc.abstractmethod
    def max_length(self) -> int: ...

    @property
    @abc.abstractmethod
    def name(self) -> str: ...


class NullPositionalEncoding(BasePositionalEncoding):
    """No-op positional encoding."""

    def __init__(self, *args, max_length: int = defaults.MAX_LENGTH, **kwargs):
        super().__init__()
        self._max_length = max_length

    def forward(
        self,
        embedded: torch.Tensor,
        symbol: torch.Tensor,
    ) -> torch.Tensor:
        return embedded

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def name(self) -> str:
        return "null"


class AbsolutePositionalEncoding(BasePositionalEncoding):
    """Absolute positional encoding.

    Each position is associated with an embedding.

    Args:
        embedding_size (int, optional).
        max_length (int, optional).
    """

    def __init__(
        self,
        embedding_size: int = defaults.EMBEDDING_SIZE,
        max_length: int = defaults.MAX_LENGTH,
    ):
        super().__init__()
        self.positional_encoding = nn.Embedding(max_length, embedding_size)

    def forward(
        self,
        embedded: torch.Tensor,
        symbol: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the positional encoding.

        Args:
            embedded (torch.Tensor): embedded sequence.
            symbol (torch.Tensor): symbol indices to encode.

        Returns:
            torch.Tensor: positional embedding.
        """
        indices = torch.arange(symbol.size(1), device=symbol.device)
        if indices.size(0) > self.max_length:
            raise Error(
                f"Sequence length {indices.size(0)} exceeds "
                f"max_length {self.max_length}"
            )
        out = self.positional_encoding(indices)
        out = out.expand(symbol.size(0), -1, -1)
        # Zeros out pads.
        out = out * symbol.ne(special.PAD_IDX).unsqueeze(2)
        return embedded + out

    @property
    def max_length(self) -> int:
        return self.positional_encoding.num_embeddings

    @property
    def name(self) -> str:
        return "absolute"


class SinusoidalPositionalEncoding(BasePositionalEncoding):
    """Sinusoidal positional encoding.

    After:
        https://pytorch.org/tutorials/beginner/transformer_tutorial.html.

    Args:
        embedding_size (int, optional).
        max_length (int, optional).
    """

    def __init__(
        self,
        embedding_size: int = defaults.EMBEDDING_SIZE,
        max_length: int = defaults.MAX_LENGTH,
    ):
        super().__init__()
        positional_encoding = torch.zeros(max_length, embedding_size)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        scale_factor = -math.log(10000.0) / embedding_size
        div_term = torch.exp(
            torch.arange(0, embedding_size, 2).float() * scale_factor
        )
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(
        self,
        embedded: torch.Tensor,
        symbol: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the positional encoding.

        Args:
            embedded (torch.Tensor): embedded sequence.
            symbol (torch.Tensor): symbol indices to encode.

        Returns:
            torch.Tensor: positional embedding.
        """
        indices = torch.arange(symbol.size(1), device=symbol.device)
        if indices.size(0) > self.max_length:
            raise Error(
                f"Sequence length {indices.size(0)} exceeds "
                f"max_length {self.max_length}"
            )
        out = self.positional_encoding[:, indices, :]
        out = out.expand(symbol.size(0), -1, -1)
        # Zeros out pads.
        out = out * symbol.ne(special.PAD_IDX).unsqueeze(2)
        return embedded + out

    @property
    def max_length(self) -> int:
        return self.positional_encoding.size(1)

    @property
    def name(self) -> str:
        return "sinusoidal"
