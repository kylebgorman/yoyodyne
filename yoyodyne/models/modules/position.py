"""Positional encoding modules for transformers."""

import abc
import math
from typing import Optional

import torch
from torch import nn

from ... import defaults, special


class BaseEncoding(abc.ABC):
    """Abstract base class for positional encodings."""

    @abc.abstractmethod
    def forward(
        self,
        embedded: torch.Tensor,
        symbol: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor: ...

    @property
    @abc.abstractmethod
    def max_length(self) -> int: ...


class NoEncoding(BaseEncoding, nn.Module):
    """No-op positional encoding."""

    def __init__(self, max_length: int = defaults.MAX_LENGTH):
        super().__init__()
        self._max_length = max_length

    def forward(
        self,
        embedded: torch.Tensor,
        symbol: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor: ...

    @property
    def max_length(self) -> int:
        return self._max_length


class AbsolutePositionalEncoding(BaseEncoding, nn.Module):
    """Absolute positional encoding.

    Args:
        d_model (int, optional).
        max_length (int, optional).
    """

    def __init__(
        self,
        d_model: int = defaults.EMBEDDING_SIZE,
        max_length: int = defaults.MAX_LENGTH,
    ):
        super().__init__()
        self.positional_encoding = nn.Embedding(max_length, d_model)

    def forward(
        self,
        embedded: torch.Tensor,
        symbol: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes the positional encoding.

        Args:
            embedded (torch.Tensor): embedded sequence.
            symbol (torch.Tensor): symbol indices to encode.
            mask (torch.Tensor, optional): optional mask for positions not to
                be encoded.

        Returns:
            torch.Tensor: positional embedding.
        """
        if mask is None:
            indices = torch.arange(symbol.size(1), device=symbol.device)
        else:
            indices = torch.cumsum(~mask, dim=1) - 1
            indices.clamp_(min=0)
        out = self.positional_encoding(indices)
        # Expands to batch size.
        out = out.unsqueeze(0).repeat(symbol.size(0), 1, 1)
        # Zeros out pads.
        out *= symbol.ne(special.PAD_IDX).unsqueeze(2)
        return embedded + out


class SinusoidalEncoding(BaseEncoding, nn.Module):
    """Sinusoidal positional encoding.

    After:
        https://pytorch.org/tutorials/beginner/transformer_tutorial.html.

    Args:
        d_model (int, optional).
        max_length (int, optional).
    """

    def __init__(
        self,
        d_model: int = defaults.EMBEDDING_SIZE,
        max_length: int = defaults.MAX_LENGTH,
    ):
        super().__init__()
        positional_encoding = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        scale_factor = -math.log(10000.0) / d_model
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * scale_factor
        )
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(
        self,
        embedded: torch.Tensor,
        symbol: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes the positional encoding.

        Args:
            embedded (torch.Tensor): embedded sequence.
            symbol (torch.Tensor): symbol indices to encode.
            mask (torch.Tensor, optional): optional mask for positions not to
                be encoded.

        Returns:
            torch.Tensor: positional embedding.
        """
        out = self.positional_encoding.repeat(symbol.size(0), 1, 1)
        if mask is None:
            indices = torch.arange(symbol.size(1), device=symbol.device)
        else:
            indices = torch.cumsum(~mask, dim=1) - 1
            indices.clamp_(min=0)
        # Selects the tensors at the specified indices.
        out = out[torch.arange(out.size(0)).unsqueeze(-1), indices]
        # Zeros out pads.
        out = out * symbol.ne(special.PAD_IDX).unsqueeze(2)
        return embedded + out

    @property
    def max_length(self) -> int:
        return self.positional_encoding.size(1)
