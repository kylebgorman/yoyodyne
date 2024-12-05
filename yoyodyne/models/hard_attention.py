"""Hard monotonic neural HMM classes."""

import argparse
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import nn

from .. import data, defaults, special
from . import modules, rnn


class HardAttentionRNNModel(rnn.RNNModel):
    """Base class for hard attention models.

    Learns probability distribution of target string by modeling transduction
    of source string to target string as Markov process. Assumes each character
    produced is conditioned by state transitions over each source character.

    Default model assumes independence between state and non-monotonic
    progression over source string. `enforce_monotonic` enforces monotonic
    state transition (model progresses over each source character), and
    `attention_context` allows conditioning of state transition over the
    previous n states.

    After:
        Wu, S. and Cotterell, R. 2019. Exact hard monotonic attention for
        character-level transduction. In _Proceedings of the 57th Annual
        Meeting of the Association for Computational Linguistics_, pages
        1530-1537.

    Original implementation: https://github.com/shijie-wu/neural-transducer

     Args:
        *args: passed to superclass.
        enforce_monotonic (bool, optional): enforces monotonic state
            transition in decoding.
        attention_context (int, optional): size of context window for
        conditioning state transition; if 0, state transitions are
            independent.
        **kwargs: passed to superclass.
    """

    enforce_monotonic: bool
    attention_context: int

    def __init__(
        self,
        *args,
        enforce_monotonic=defaults.ENFORCE_MONOTONIC,
        attention_context=defaults.ATTENTION_CONTEXT,
        **kwargs,
    ):
        self.enforce_monotonic = enforce_monotonic
        self.attention_context = attention_context
        super().__init__(*args, **kwargs)
        self.classifier = nn.Linear(
            self.decoder.output_size, self.target_vocab_size
        )
        assert (
            self.teacher_forcing
        ), "Teacher forcing disabled but required by this model"

    def beam_decode(self, *args, **kwargs):
        """Overrides incompatible implementation inherited from RNNModel."""
        raise NotImplementedError(
            f"Beam search not implemented for {self.name} model"
        )

    def decode(
        self,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decodes a sequence given the encoded input.

        Decodes until all sequences in a batch have reached END up to length of
        `target` args.

        Args:
            encoder_output (torch.Tensor): batch of encoded input symbols
                of shape batch_size x src_len x (encoder_hidden *
                num_directions).
            encoder_mask (torch.Tensor): mask for the batch of encoded
                input symbols of shape batch_size x src_len.
            target (torch.Tensor): target symbols, decodes up to
                `len(target)` symbols.

        Returns:
            Tuple[torch.Tensor,torch.Tensor]: emission probabilities
                for each state (target symbol) of shape target_len x
                batch_size x src_len x vocab_size, and transition
                probabilities for each state (target symbol) of shape
                target_len x batch_size x src_len x src_len.
        """
        batch_size = encoder_mask.shape[0]
        emissions = []
        transitions = []
        # Like in simpler RNN models, the input to the underlying decoder
        # module is `modules.ModuleOutput`; unlike those models, the output is
        # the custom `modulesHardAttentionOutput`.
        decoder_input = self.initial_input(batch_size)
        decoder_output = self.decode_step(
            decoder_input, encoder_output, encoder_mask
        )
        emissions.append(decoder_output.output)
        transitions.append(decoder_output.embedded)
        for target_symbol_idx in range(target.size(1)):
            decoder_input = modules.ModuleOutput(
                target[:, target_symbol_idx].unsqueeze(-1),
                decoder_output.hidden,
                decoder_output.cell,
            )
            decoder_output = self.decode_step(
                decoder_input, encoder_output, encoder_mask
            )
            emissions.append(decoder_output.emissions)
            transitions.append(decoder_output.transitions)
        return torch.stack(emissions), torch.stack(transitions)

    def decode_step(
        self,
        decoder_input: modules.ModuleOutput,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> modules.HardAttentionModuleOutput:
        """Performs a single decoding step.

        Args:
            decoder_input (modules.ModuleOutput).
            encoder_output (torch.Tensor): batch of encoded source symbols.
            encoder_mask (torch.Tensor): source symbol mask.

        Returns:
            modules.ModuleOutput: emissions, hidden (and where appropriate) cell
                states, and transitions.
        """
        decoder_output = self.decoder(
            decoder_input, encoder_output, encoder_mask
        )
        # Emissions are now logits.
        emissions = self.classifier(decoder_output.emissions)
        # Emissions are now log probabilities.
        emissions = torch.nn.functional.log_softmax(emissions, dim=-1)
        transitions = decoder_output.transitions
        if self.enforce_monotonic:
            transitions = self._apply_mono_mask(transitions)
        return modules.HardAttentionModuleOutput(
            emissions, transitions, decoder_output.hidden, decoder_output.cell
        )

    def greedy_decode(
        self,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decodes a sequence given the encoded input.

        Decodes until all sequences in a batch have reached END up to a
        specified length depending on the `target` args.

        Args:
            encoder_output (torch.Tensor): batch of encoded input symbols
                of shape B x src_len x (encoder_hidden * num_directions).
            encoder_mask (torch.Tensor): mask for the batch of encoded
                input symbols of shape B x src_len.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: predictions of shape B
                x pred_seq_len, final likelihoods per prediction step of shape
                B x 1 x src_len.
        """
        batch_size = encoder_mask.shape[0]
        decoder_input = self.initial_input(batch_size)
        decoder_output = self.decode_step(
            decoder_input, encoder_output, encoder_mask
        )
        likelihood = decoder_output.transitions[:, 0].unsqueeze(1)
        predictions = self._greedy_step(decoder_output.emissions, likelihood)
        finished = torch.zeros(
            batch_size, dtype=torch.bool, device=self.device
        ) .unsqueeze(-1)
        for t in range(self.max_target_length):
            decoder_input = modules.ModuleOutput(
                predictions[:, t].unsqueeze(-1),
                decoder_output.hidden,
                decoder_output.cell,
            )
            decoder_output = self.decode_step(
                decoder_input, encoder_output, encoder_mask
            )
            print("likelihood:", likelihood.shape)
            print("transitions:", decoder_output.transitions.transpose(1, 2).shape)
            likelihood += decoder_output.transitions.transpose(1, 2)
            likelihood = likelihood.logsumexp(dim=-1, keepdim=True).transpose(
                1, 2
            )
            prediction = self._greedy_step(
                decoder_output.emissions, likelihood
            )
            finished = torch.logical_or(
                finished, (predictions == special.END_IDX)
            )
            if finished.all():
                break
            predictions = torch.cat((predictions, prediction), dim=-1)
            # prediction = torch.where(~finished, prediction, special.PAD_IDX)
            likelihoods += self._gather_at_idx(emissions, prediction)
        return predictions, likelihood

    @staticmethod
    def _greedy_step(
        emissions: torch.Tensor, likelihood: torch.Tensor
    ) -> torch.Tensor:
        """Greedily decodes current timestep.

        Args:
            emissions (torch.Tensor).
            likelihood (torch.Tensor).

        Returns:
            torch.Tensor: predictions for current timestep.
        """
        predictions = likelihood + emissions.transpose(1, 2)
        predictions = predictions.logsumexp(dim=-1)
        _, predictions = torch.max(predictions, dim=-1, keepdim=True)
        return predictions

    @staticmethod
    def _gather_at_idx(
        prob: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Collects probability of target index across all states in prob.

        To calculate the final emission probability, the pseudo-HMM
        graph needs to aggregate the final emission probabilities of
        target char across all potential hidden states in prob.

        Args:
            prob (torch.Tensor): log probabilities of emission states of shape
                B x src_len x vocab_size.
            target (torch.Tensor): target symbol to poll probabilities for
                shape B.

        Returns:
            torch.Tensor: emission probabilities of target symbol for each hidden
                state of size B 1 x src_len.
        """
        batch_size, src_seq_len, _ = prob.shape
        idx = target.view(-1, 1).expand(batch_size, src_seq_len).unsqueeze(-1)
        output = torch.gather(prob, -1, idx).view(batch_size, 1, src_seq_len)
        idx = idx.view(batch_size, 1, src_seq_len)
        pad_mask = (idx != special.PAD_IDX).float()
        return output * pad_mask

    @staticmethod
    def _apply_mono_mask(
        transition_prob: torch.Tensor,
    ) -> torch.Tensor:
        """Applies monotonic attention mask to transition probabilities.

        Enforces a 0 log-probability values for all non-monotonic relations
        in the transition_prob tensor (i.e., all values i < j per row j).

        Args:
            transition_prob (torch.Tensor): transition probabilities between
                all hidden states (source sequence) of shape
                B x src_len x src_len.

        Returns:
            torch.Tensor: masked transition probabilities of shape
                B x src_len x src_len.
        """
        mask = torch.ones_like(transition_prob[0]).triu().unsqueeze(0)
        # Using 0 log-probability value for masking; this is borrowed from the
        # original implementation.
        mask = (mask - 1) * defaults.NEG_LOG_EPSILON
        transition_prob = transition_prob + mask
        transition_prob = transition_prob - transition_prob.logsumexp(
            -1, keepdim=True
        )
        return transition_prob

    def forward(
        self,
        batch: data.PaddedBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs the encoder-decoder model.

        Args:
            batch (data.PaddedBatch).

        Returns:
            Tuple[torch.Tensor,torch.Tensor]: emission probabilities for
                each transition state of shape
                target_len x B x src_len x vocab_size, and transition
                probabilities for each transition

        Raises:
            NotImplementedError: beam search not implemented.
        """
        encoder_output = self.source_encoder(batch.source).output
        if self.has_features_encoder:
            encoder_output = self._combine_source_and_features_encoder_output(
                encoder_output,
                self.features_encoder(batch.features).output,
            )
        if self.beam_width > 1:
            # Will raise a NotImplementedError.
            return self.beam_decode(encoder_output, batch.source.mask)
        else:
            return self.greedy_decode(encoder_output, batch.source.mask)

    def _combine_source_and_features_encoder_output(
        self,
        source_encoder_output: torch.Tensor,
        features_encoder_output: torch.Tensor,
    ) -> torch.Tensor:
        """Combines source and features encodings.

        Args:
            source_encoder_output (torch.Tensor),
            features_encoder_output (torch.Tensor),

        Returns:
            torch.Tensor.
        """
        features_encoder_output = features_encoder_output.sum(
            dim=1, keepdim=True
        )
        features_encoder_output = features_encoder_output.expand(
            -1, encoder_out.shape[1], -1
        )
        return torch.cat(
            (source_encoder_output, features_encoder_output), dim=-1
        )

    def training_step(
        self, batch: data.PaddedBatch, batch_idx: int
    ) -> torch.Tensor:
        """Runs one step of training.

        This is called by the PL Trainer.

        Args:
            batch (data.PaddedBatch)
            batch_idx (int).

        Returns:
            torch.Tensor: loss.
        """
        # Forward pass produces loss by default.
        emissions, transitions = self(batch)
        loss = self.loss_func(batch.target.padded, emissions, transitions)
        self.log(
            "train_loss",
            loss,
            batch_size=len(batch),
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch: data.PaddedBatch, batch_idx: int) -> Dict:
        predictions, likelihood = self(batch)
        # Processes for accuracy calculation.
        val_eval_items_dict = {}
        for evaluator in self.evaluators:
            final_predictions = evaluator.finalize_predictions(predictions)
            final_golds = evaluator.finalize_golds(batch.target.padded)
            val_eval_items_dict[evaluator.name] = evaluator.get_eval_item(
                final_predictions, final_golds
            )
        val_eval_items_dict.update({"val_loss": -likelihood.mean()})
        return val_eval_items_dict

    def predict_step(self, batch: data.PaddedBatch, batch_idx: int) -> Dict:
        predictions, _ = self(batch)
        return predictions

    def _get_loss_func(
        self,
    ) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """Returns the actual function used to compute loss.

        Returns:
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]: configured
                loss function.
        """
        return self._loss

    def _loss(
        self,
        target: torch.Tensor,
        log_probs: torch.Tensor,
        transition_probs: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: Currently we're storing a concatenation of loss tensors for
        # each time step. This is costly. Revisit this calculation and see if
        # we can use DP to simplify.
        fwd = transition_probs[0, :, 0].unsqueeze(1) + self._gather_at_idx(
            log_probs[0], target[:, 0]
        )
        for target_char_idx in range(1, target.shape[1]):
            fwd = fwd + transition_probs[target_char_idx].transpose(1, 2)
            fwd = fwd.logsumexp(dim=-1, keepdim=True).transpose(1, 2)
            fwd = fwd + self._gather_at_idx(
                log_probs[target_char_idx], target[:, target_char_idx]
            )
        loss = -torch.logsumexp(fwd, dim=-1).mean() / target.shape[1]
        return loss

    # Interface.

    def get_decoder(self):
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError

    @staticmethod
    def add_argparse_args(parser: argparse.ArgumentParser) -> None:
        """Adds HMM configuration options to the argument parser.

        Args:
            parser (argparse.ArgumentParser).
        """
        parser.add_argument(
            "--enforce_monotonic",
            action="store_true",
            default=defaults.ENFORCE_MONOTONIC,
            help="Enforce monotonicity "
            "(hard attention architectures only). Default: %(default)s.",
        )
        parser.add_argument(
            "--no_enforce_monotonic",
            action="store_false",
            dest="enforce_monotonic",
        )
        parser.add_argument(
            "--attention_context",
            type=int,
            default=defaults.ATTENTION_CONTEXT,
            help="Width of attention context "
            "(hard attention architectures only). Default: %(default)s.",
        )


class HardAttentionGRUModel(HardAttentionRNNModel, rnn.GRUModel):
    """Hard attention with GRU backend."""

    def get_decoder(self):
        if self.attention_context > 0:
            return modules.ContextHardAttentionGRUDecoder(
                attention_context=self.attention_context,
                bidirectional=False,
                decoder_input_size=(
                    self.source_encoder.output_size
                    + self.features_encoder.output_size
                    if self.has_features_encoder
                    else self.source_encoder.output_size
                ),
                dropout=self.dropout,
                embeddings=self.embeddings,
                embedding_size=self.embedding_size,
                hidden_size=self.hidden_size,
                layers=self.decoder_layers,
                num_embeddings=self.target_vocab_size,
            )
        else:
            return modules.HardAttentionGRUDecoder(
                bidirectional=False,
                decoder_input_size=(
                    self.source_encoder.output_size
                    + self.features_encoder.output_size
                    if self.has_features_encoder
                    else self.source_encoder.output_size
                ),
                dropout=self.dropout,
                embedding_size=self.embedding_size,
                embeddings=self.embeddings,
                hidden_size=self.hidden_size,
                layers=self.decoder_layers,
                num_embeddings=self.target_vocab_size,
            )

    @property
    def name(self) -> str:
        return "hard attention GRU"


class HardAttentionLSTMModel(HardAttentionRNNModel, rnn.LSTMModel):
    """Hard attention with LSTM backend."""

    def get_decoder(self):
        if self.attention_context > 0:
            return modules.ContextHardAttentionLSTMDecoder(
                attention_context=self.attention_context,
                bidirectional=False,
                decoder_input_size=(
                    self.source_encoder.output_size
                    + self.features_encoder.output_size
                    if self.has_features_encoder
                    else self.source_encoder.output_size
                ),
                dropout=self.dropout,
                hidden_size=self.hidden_size,
                embeddings=self.embeddings,
                embedding_size=self.embedding_size,
                layers=self.decoder_layers,
                num_embeddings=self.target_vocab_size,
            )
        else:
            return modules.HardAttentionLSTMDecoder(
                bidirectional=False,
                decoder_input_size=(
                    self.source_encoder.output_size
                    + self.features_encoder.output_size
                    if self.has_features_encoder
                    else self.source_encoder.output_size
                ),
                dropout=self.dropout,
                embeddings=self.embeddings,
                embedding_size=self.embedding_size,
                hidden_size=self.hidden_size,
                layers=self.decoder_layers,
                num_embeddings=self.target_vocab_size,
            )

    @property
    def name(self) -> str:
        return "hard attention LSTM"
