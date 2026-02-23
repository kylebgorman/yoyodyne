"""Positional encoding modules for transformers."""

import math
from typing import Optional

import torch
from torch import nn

from ... import defaults, special


class PositionalEncoding(nn.Module):
    """Positional encoding.

    After:
        https://pytorch.org/tutorials/beginner/transformer_tutorial.html.

    Args:
        d_model (int, optional).
        max_source_length (int, optional).
    """

    def __init__(
        self,
        d_model: int = defaults.EMBEDDING_SIZE,
        max_source_length: int = defaults.MAX_LENGTH,
    ):
        super().__init__()
        positional_encoding = torch.zeros(max_source_length, d_model)
        position = torch.arange(
            0, max_source_length, dtype=torch.float
        ).unsqueeze(1)
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
        symbol: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes the positional encoding.

        Args:
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
        # FIXME
        length = symbol.size(1)
        assert (
            length <= self.max_length
        ), f"Symbol length {length} out of bounds (> {self.max_length})"
        # /FIXME
        # Selects the tensors from `out` at the specified indices.
        out = out[torch.arange(out.size(0)).unsqueeze(-1), indices]
        # Zeros out pads.
        out = out * symbol.ne(special.PAD_IDX).unsqueeze(2)
        return out

    @property
    def max_length(self) -> int:
        return self.positional_encoding.size(1)
