"""RNN model classes."""

import argparse
from typing import Optional, Tuple, Union

import torch
from torch import nn

from .. import data, defaults, special
from . import base, beam_search, embeddings, modules


class RNNModel(base.BaseModel):
    """Base class for RNN models.

    In lieu of attention, we concatenate the last (non-padding) hidden state of
    the encoder to the decoder hidden state.
    """

    # Constructed inside __init__.
    classifier: nn.Linear
    h0: nn.Parameter

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier = nn.Linear(self.hidden_size, self.target_vocab_size)
        self.h0 = nn.Parameter(torch.rand(self.hidden_size))

    def init_embeddings(
        self,
        num_embeddings: int,
        embedding_size: int,
    ) -> nn.Embedding:
        """Initializes the embedding layer.

        Args:
            num_embeddings (int): number of embeddings.
            embedding_size (int): dimension of embeddings.

        Returns:
            nn.Embedding: embedding layer.
        """
        return embeddings.normal_embedding(num_embeddings, embedding_size)

    def initial_input(self, batch_size: int) -> modules.ModuleOutput:
        """The decoder input at the start of decoding."""
        raise NotImplementedError

    def _initial_symbol(self, batch_size: int) -> torch.Tensor:
        """The initial symbol.

        Args:
            batch_size (int).

        Returns:
            torch.Tensor: initial symbol.
        """
        return (
            torch.tensor([special.START_IDX], device=self.device)
            .repeat(batch_size)
            .unsqueeze(1)
        )

    def _initial_hidden(self, batch_size: int) -> torch.Tensor:
        """The initial hidden state.

        We treat this as a model parameter.

        Args:
            batch_size (int).

        Returns:
            torch.Tensor: initial hidden state.
        """
        return self.h0.repeat(self.decoder_layers, batch_size, 1)

    def beam_decode(
        self,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decodes with a beam.

        Args:
            encoder_output (torch.Tensor): batch of encoded source symbols.
            encoder_mask (torch.Tensor): source symbol mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: predictions and the
                associated log-likelihoods.
        """
        # This currently assumes batch size is 1.
        batch_size = encoder_mask.shape[0]
        # Misleadingly named to abuse the GC.
        decoder_output = self.initial_input(batch_size)
        beam = beam_search.Beam.from_initial_input(decoder_output)
        for t in range(self.max_target_length):
            with beam_search.BeamHelper(self.beam_width) as helper:
                for node in beam.nodes:
                    # Misleadingly named to abuse the GC.
                    decoder_output = node.decoder_input().to(self.device)
                    decoder_output = self.decoder(
                        decoder_output, encoder_output, encoder_mask
                    )
                    # Overwrites the raw decoder outputs with logits, needed
                    # to generate next-symbol predictions.
                    decoder_output.output = self.classifier(
                        decoder_output.output
                    ).squeeze((0, 1))
                    helper.record(node, decoder_output)
            beam = helper.beam()
            if beam.is_final:
                break
        # --> B x beam_width x seq_length.
        predictions = (
            nn.utils.rnn.pad_sequence(
                [
                    torch.tensor(cell.symbols, device=self.device)
                    for cell in beam.nodes
                ],
                padding_value=special.PAD_IDX,
            )
            .unsqueeze(0)
            .transpose(1, 2)
        )
        loglikes = torch.tensor(
            [cell.loglike for cell in beam.nodes],
            device=self.device,
        ).unsqueeze(0)
        return predictions, loglikes

    def greedy_decode(
        self,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor,
        teacher_forcing: bool,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decodes a sequence given the encoded input.

        Decodes until all sequences in a batch have reached END up to a
        specified length depending on the `target` args.

        Args:
            encoder_output (torch.Tensor): batch of encoded source symbols.
            encoder_mask (torch.Tensor): source symbol mask.
            teacher_forcing (bool): Whether or not to decode with teacher
                forcing.
            target (torch.Tensor, optional): target symbols; we decode up to
                `len(target)` symbols. If omitted, we decode up to
                `self.max_target_length` symbols.

        Returns:
            torch.Tensor: tensor of predictions of shape seq_len x
                batch_size x target_vocab_size.
        """
        batch_size = encoder_mask.shape[0]
        # Misleadingly named to abuse the GC.
        decoder_output = self.initial_input(batch_size)
        predictions = []
        max_target_length = (
            target.size(1) if target is not None else self.max_target_length
        )
        # Tracks when each sequence has decoded an END; not needed when using
        # teacher forcing.
        finished = (
            None
            if teacher_forcing
            else torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        )
        for t in range(max_target_length):
            decoder_output = self.decoder(
                decoder_output, encoder_output, encoder_mask
            )
            logits = self.classifier(decoder_output.output)
            predictions.append(logits.squeeze(1))
            if teacher_forcing:
                # Inserts the gold "teacher" predictions.
                decoder_output.output = target[:, t].unsqueeze(1)
            else:
                # Inserts the "student" predictions.
                decoder_output.output = self._get_predicted(logits)
                finished = torch.logical_or(
                    finished, (decoder_output.output == special.END_IDX)
                )
                if finished.all():
                    # Breaks when all sequences have predicted an END symbol.
                    # If we have a target (and are thus computing loss), we
                    # only break once we have decoded at least the the same
                    # number of steps as the target length.
                    if target is None or (
                        decoder_output.output.size(-1) >= target.size(-1)
                    ):
                        break
        return torch.stack(predictions).transpose(0, 1)

    def forward(
        self,
        batch: data.PaddedBatch,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Runs the encoder-decoder model.

        Args:
            batch (data.PaddedBatch).

        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
                for beam decoding, a tuple of predictions and unnormalized log
                likelihoods for each prediction; for greedy decoding, the
                predictions tensor.
        """
        encoder_output = self.source_encoder(batch.source).output
        # This method has a polymorphic return type because beam search needs
        # to return the log-likelihoods too.
        if self.beam_width > 1:
            return self.beam_decode(
                encoder_output,
                batch.source.mask,
            )
        else:
            return self.greedy_decode(
                encoder_output,
                batch.source.mask,
                self.teacher_forcing if self.training else False,
                batch.target.padded if batch.target else None,
            )

    @staticmethod
    def add_argparse_args(parser: argparse.ArgumentParser) -> None:
        """Adds RNN configuration options to the argument parser.

        Args:
            parser (argparse.ArgumentParser).
        """
        parser.add_argument(
            "--bidirectional",
            action="store_true",
            default=defaults.BIDIRECTIONAL,
            help="Uses a bidirectional encoder (RNN-backed architectures "
            "only. Default: enabled.",
        )
        parser.add_argument(
            "--no_bidirectional",
            action="store_false",
            dest="bidirectional",
        )

    # Interface.

    def get_decoder(self):
        raise NotImplementedError

    def initial_input(self, batch_size: int) -> modules.ModuleOutput:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError


class GRUModel(RNNModel):
    """GRU encoder-decoder without attention."""

    def get_decoder(self) -> modules.GRUDecoder:
        return modules.GRUDecoder(
            bidirectional=False,
            decoder_input_size=self.source_encoder.output_size,
            dropout=self.dropout,
            embedding_size=self.embedding_size,
            embeddings=self.embeddings,
            hidden_size=self.hidden_size,
            layers=self.decoder_layers,
            num_embeddings=self.vocab_size,
        )

    def initial_input(self, batch_size: int) -> modules.ModuleOutput:
        """The decoder input at the start of decoding.

        Args:
            batch_size (int).

        Returns:
            modules.ModuleOutput.
        """
        symbol = self._initial_symbol(batch_size)
        hidden = self._initial_hidden(batch_size)
        return modules.ModuleOutput(symbol, hidden)

    @property
    def name(self) -> str:
        return "GRU"


class LSTMModel(RNNModel):
    """LSTM encoder-decoder without attention.

    Args:
        *args: passed to superclass.
        **kwargs: passed to superclass.
    """

    # Implementationally this is identical to the GRU except the presence of
    # the cell state and the initial cell state parameter.

    c0: nn.Parameter

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c0 = nn.Parameter(torch.rand(self.hidden_size))

    def get_decoder(self) -> modules.LSTMDecoder:
        return modules.LSTMDecoder(
            bidirectional=False,
            decoder_input_size=self.source_encoder.output_size,
            dropout=self.dropout,
            embedding_size=self.embedding_size,
            embeddings=self.embeddings,
            hidden_size=self.hidden_size,
            layers=self.decoder_layers,
            num_embeddings=self.vocab_size,
        )

    def initial_input(self, batch_size: int) -> modules.ModuleOutput:
        """The decoder input at the start of decoding.

        Args:
            batch_size (int).

        Returns:
            modules.ModuleOutput.
        """
        symbol = self._initial_symbol(batch_size)
        hidden = self._initial_hidden(batch_size)
        cell = self._initial_cell(batch_size)
        return modules.ModuleOutput(symbol, hidden, cell)

    def _initial_cell(self, batch_size: int) -> torch.Tensor:
        """The initial cell state.

        We treat this as a model parameter.

        Args:
            batch_size (int).

        Returns:
            torch.Tensor: initial cell state.
        """
        return self.c0.repeat(self.decoder_layers, batch_size, 1)

    @property
    def name(self) -> str:
        return "LSTM"


class AttentiveGRUModel(GRUModel):
    """GRU encoder-decoder with attention."""

    def get_decoder(self) -> modules.AttentiveGRUDecoder:
        return modules.AttentiveGRUDecoder(
            attention_input_size=self.source_encoder.output_size,
            bidirectional=False,
            decoder_input_size=self.source_encoder.output_size,
            dropout=self.dropout,
            embeddings=self.embeddings,
            embedding_size=self.embedding_size,
            hidden_size=self.hidden_size,
            layers=self.decoder_layers,
            num_embeddings=self.vocab_size,
        )

    @property
    def name(self) -> str:
        return "attentive GRU"


class AttentiveLSTMModel(LSTMModel):
    """LSTM encoder-decoder with attention."""

    def get_decoder(self) -> modules.AttentiveLSTMDecoder:
        return modules.AttentiveLSTMDecoder(
            attention_input_size=self.source_encoder.output_size,
            bidirectional=False,
            decoder_input_size=self.source_encoder.output_size,
            dropout=self.dropout,
            embeddings=self.embeddings,
            embedding_size=self.embedding_size,
            hidden_size=self.hidden_size,
            layers=self.decoder_layers,
            num_embeddings=self.vocab_size,
        )

    @property
    def name(self) -> str:
        return "attentive LSTM"
