"""Transducer model class."""

from typing import Callable, Dict, List, Optional, Tuple

import numpy
import torch
from maxwell import actions
from torch import nn

from .. import data, defaults, special, util
from . import expert, modules, rnn


class TransducerRNNModel(rnn.RNNModel):
    """Base class for transducer models.

    This uses a trained oracle for imitation learning edits.

    After:
        Makarov, P., and Clematide, S. 2018. Imitation learning for neural
        morphological string transduction. In Proceedings of the 2018
        Conference on Empirical Methods in Natural Language Processing, pages
        2877–2882.

     Args:
        expert (expert.Expert): oracle that guides training for transducer.
        *args: passed to superclass.
        **kwargs: passed to superclass.
    """

    expert: expert.Expert

    def __init__(
        self,
        expert,
        *args,
        **kwargs,
    ):
        """Initializes transducer model.

        Args:
            expert (expert.Expert): oracle that guides training for transducer.
            *args: passed to superclass.
            **kwargs: passed to superclass.
        """
        super().__init__(*args, **kwargs)
        self.expert = expert

    # Properties.

    @property
    def decoder_input_size(self) -> int:
        if self.has_features_encoder:
            return (
                self.source_encoder.output_size
                + self.features_encoder.output_size
            )
        else:
            return self.source_encoder.output_size

    @property
    def beg_idx(self) -> int:
        return self.expert.actions.beg_idx

    @property
    def copy_idx(self) -> int:
        return self.expert.actions.copy_idx

    @property
    def del_idx(self) -> int:
        return self.expert.actions.del_idx

    @property
    def end_idx(self) -> int:
        return self.expert.actions.end_idx

    @property
    def insertions(self) -> List[Tuple[int, actions.ConditionalEdit]]:
        return self.expert.actions.insertions

    @property
    def substitutions(self) -> List[Tuple[int, actions.ConditionalEdit]]:
        return self.expert.actions.substitutions

    @property
    def vocab_offset(self) -> int:
        return self.vocab_size - self.target_vocab_size

    # Implemented interface.

    def forward(
        self,
        batch: data.PaddedBatch,
    ) -> Tuple[List[List[int]], torch.Tensor]:
        """Runs the encoder-decoder model.

        Args:
            batch (data.PaddedBatch).

        Returns:
            Tuple[List[List[int]], torch.Tensor]: encoded prediction values
                and loss tensor; due to transducer setup, prediction is
                performed during training, so these are returned.
        """
        encoder_output = self.source_encoder(batch.source).output
        # Ignores start symbol.
        encoder_output = encoder_output[:, 1:, :]
        source = batch.source.padded[:, 1:]
        source_mask = batch.source.mask[:, 1:]
        if self.has_features_encoder:
            encoder_output = self._combine_source_and_features_encoder_output(
                encoder_output,
                self.features_encoder(batch.features).output,
            )
        if self.beam_width > 1:
            # Will raise a NotImplementedError.
            return self.beam_decode(
                encoder_output,
                source,
                source_mask,
                teacher_forcing=(
                    self.teacher_forcing if self.training else False
                ),
                target=batch.target.padded if batch.target else None,
                target_mask=batch.target.mask if batch.target else None,
            )
        else:
            return self.greedy_decode(
                encoder_output,
                source,
                source_mask,
                teacher_forcing=(
                    self.teacher_forcing if self.training else False
                ),
                target=batch.target.padded if batch.target else None,
                target_mask=batch.target.mask if batch.target else None,
            )

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
        features_encoder_output = features_encoder_output.mean(
            dim=1, keepdim=True
        )
        return torch.cat(
            (
                source_encoder_output,
                features_encoder_output.expand(
                    -1, source_encoder_output.shape[1], -1
                ),
            ),
            dim=2,
        )

    def beam_decode(self, *args, **kwargs):
        """Overrides incompatible implementation inherited from RNNModel."""
        raise NotImplementedError(
            f"Beam search not implemented for {self.name} model"
        )

    def greedy_decode(
        self,
        encoder_output: torch.Tensor,
        source,
        source_mask: torch.Tensor,
        teacher_forcing: bool,
        target: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[List[List[int]], torch.Tensor]:
        """Decodes a sequence given the encoded input.

        This essentially serves as a wrapper for looping decode_step.

        Args:
            encoder_output (torch.Tensor): encoded source symbols.
            source (torch.Tensor): source symbols.
            source_mask (torch.Tensor): source symbol mask.
            teacher_forcing (bool): whether or not to decode with teacher
                forcing; determines whether or not to roll out optimal actions.
            target (torch.Tensor, optional): target symbols.
            target_mask (torch.Tensor, optional): target symbol mask.

        Returns:
            Tuple[List[List[int]], torch.Tensor]: encoded prediction values
                and loss tensor; due to transducer setup, prediction is
                performed during training, so these are returned.
        """
        batch_size = source_mask.shape[0]
        source_length = (~source_mask).sum(dim=1)
        alignment = torch.zeros(
            batch_size, device=self.device, dtype=torch.int64
        )
        action_count = torch.zeros_like(alignment)
        last_action = torch.full(
            (batch_size,), self.beg_idx, device=self.device
        )
        loss = torch.zeros(batch_size, device=self.device)
        predictions = [[] for _ in range(batch_size)]
        # Converting encodings for prediction.
        if target is not None:
            # Target and source need to be integers for SED values.
            # Clips END and PAD from source and target.
            source = [
                s[~smask].tolist()[:-1]
                for s, smask in zip(source, source_mask)
            ]
            target = [
                t[~tmask].tolist()[:-1]
                for t, tmask in zip(target, target_mask)
            ]
        # Misleadingly named to abuse the GC.
        decoder_output = self.initial_input(batch_size)
        for _ in range(self.max_target_length):
            # Checks if completed all sequences.
            not_complete = last_action != self.end_idx
            if not not_complete.any():
                # I.e., if all are complete.
                break
            # Proceeds to make new edit; new action for all current decoding.
            action_count = torch.where(
                not_complete,
                action_count + 1,
                action_count,
            )
            # We offset the action idx by the symbol vocab size so that we
            # can index into the shared embeddings matrix.
            decoder_output.output = (
                last_action.unsqueeze(1) + self.vocab_offset
            )
            decoder_output = self.decoder(
                decoder_output,
                encoder_output,
                # Accomodates RNNDecoder; see encoder_mask behavior.
                ~(alignment.unsqueeze(1) + 1),
            )
            logits = self.classifier(decoder_output.output).squeeze(1)
            # If given targets, asks expert for optimal actions.
            optimal_actions = (
                self._batch_expert_rollout(
                    source,
                    target,
                    alignment,
                    predictions,
                    not_complete,
                )
                if target is not None
                else None
            )
            last_action = self._decode_action_step(
                logits,
                alignment,
                source_length,
                not_complete,
                optimal_actions if teacher_forcing else None,
            )
            alignment = self._update_prediction(
                last_action, source, alignment, predictions
            )
            # If target, validation or training step loss required.
            if target is not None:
                log_sum_loss = self._log_sum_softmax_loss(
                    logits, optimal_actions
                )
                loss = torch.where(not_complete, log_sum_loss + loss, loss)
        loss = -torch.mean(loss / action_count)
        return predictions, loss

    def _decode_action_step(
        self,
        logits: torch.Tensor,
        alignment: torch.Tensor,
        input_length: torch.Tensor,
        not_complete: torch.Tensor,
        optimal_actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decodes logits to find edit action.

        Finds possible actions given remaining size of source input and masks
        logits for edit action decoding.

        Args:
            logits (torch.Tensor): logit values from decode_step of shape
                B x num_actions.
            alignment (torch.Tensor): index of encoding symbols for decoding,
                per item in batch of shape B x seq_len.
            input_length (torch.Tensor): length of each item in batch.
            not_complete (torch.Tensor): boolean values designating which items
                have not terminated edits.
            optimal_actions (List[List[int]], optional): optimal actions
                determined by expert, present when loss is being calculated.

        Returns:
            torch.Tensor: chosen edit action.
        """
        # Finds valid actions given remaining input length.
        end_of_input = (input_length - alignment) <= 1  # 1 -> Last char.
        valid_actions = [
            (self._compute_valid_actions(eoi) if nc else [self.end_idx])
            for eoi, nc in zip(end_of_input, not_complete)
        ]
        # Masks invalid actions.
        logits = self._action_probability_mask(logits, valid_actions)
        return self._choose_action(logits, not_complete, optimal_actions)

    def _compute_valid_actions(self, end_of_input: bool) -> List[int]:
        """Gives all possible actions for remaining length of edits.

        Args:
            end_of_input (bool): indicates if this is the last input from
                string; if true, only insertions are available.

        Returns:
            List[actions.Edit]: actions known by transducer.
        """
        valid_actions = [self.end_idx]
        valid_actions.extend(self.insertions)
        if not end_of_input:
            valid_actions.append(self.copy_idx)
            valid_actions.append(self.del_idx)
            valid_actions.extend(self.substitutions)
        return valid_actions

    def _action_probability_mask(
        self, logits: torch.Tensor, valid_actions: List[int]
    ) -> torch.Tensor:
        """Masks non-valid actions in logits."""
        with torch.no_grad():
            mask = torch.full_like(logits, defaults.NEG_INF)
            for row, action in zip(mask, valid_actions):
                row[action] = 0.0
            logits = mask + logits
        return logits

    def _choose_action(
        self,
        logits: torch.Tensor,
        not_complete: torch.Tensor,
        optimal_actions: Optional[List[List[int]]] = None,
    ) -> torch.Tensor:
        """Chooses transducer action from log_prob distribution.

        If training, uses dynamic oracle for selection.

        Args:
            log_probs (torch.Tensor): probability distribution of actions.
            not_complete (torch.Tensor): boolean tensor of batch length to
                indicate if each item in batch is complete.
            optimal_actions (Optional[List[List[int]]]): optional encoded
                actions to use for action selection.

        Returns:
            torch.Tensor: action encodings.
        """
        # TODO: Merge logic into PyTorch methods.
        log_probs = nn.functional.log_softmax(logits, dim=1)
        if optimal_actions is None:
            # Argmax decoding.
            next_action = [
                (torch.argmax(probs, dim=0) if nc else self.end_idx)
                for probs, nc in zip(log_probs, not_complete)
            ]
        else:
            # Training with dynamic oracle; chooses from optimal actions.
            with torch.no_grad():
                if self.expert.explore():
                    # Action is picked by random exploration.
                    next_action = [
                        (self._sample(probs) if nc else self.end_idx)
                        for probs, nc in zip(log_probs, not_complete)
                    ]
                else:
                    # Action is picked from optimal_actions.
                    next_action = []
                    for action, probs, nc in zip(
                        optimal_actions, log_probs, not_complete
                    ):
                        if nc:
                            optimal_logs = probs[action]
                            idx = torch.argmax(optimal_logs, dim=0).item()
                            next_action.append(action[idx])
                        else:  # Already complete, so skip.
                            next_action.append(self.end_idx)
        return torch.tensor(next_action, device=self.device)

    # TODO: Merge action classes to remove need for this method.

    @staticmethod
    def _remap_actions(
        action_scores: Dict[actions.Edit, float]
    ) -> Dict[actions.Edit, float]:
        """Maps generative oracle's edit to conditional counterpart.

        Oracle edits are a distinct subclass from edits learned from samples.

        This will eventually be removed.

        Args:
            action_scores (Dict[actions.Edit, float]): weights for each action.

        Returns:
            Dict[actions.Edit, float]: edit action-weight pairs.
        """
        remapped_action_scores = {}
        for action, score in action_scores.items():
            if isinstance(action, actions.GenerativeEdit):
                remapped_action = action.conditional_counterpart()
            elif isinstance(action, actions.Edit):
                remapped_action = action
            else:
                raise expert.ActionError(
                    f"Unknown action: {action}, {score}, "
                    f"action_scores: {action_scores}"
                )
            remapped_action_scores[remapped_action] = score
        return remapped_action_scores

    def _expert_rollout(
        self,
        source: List[int],
        target: List[int],
        alignment: int,
        prediction: List[int],
    ) -> List[int]:
        """Rolls out with optimal expert policy.

        Args:
            source (List[int]): input string.
            target (List[int]): target string.
            alignment (int): position in source to edit.
            prediction (List[str]): current prediction.

        Returns:
            List[int]: optimal action encodings.
        """
        raw_action_scores = self.expert.score(
            source,
            target,
            alignment,
            prediction,
            max_action_seq_len=self.max_target_length,
        )
        action_scores = self._remap_actions(raw_action_scores)
        optimal_value = min(action_scores.values())
        optimal_action = sorted(
            [
                self.expert.actions.encode_unseen_action(action)
                for action, value in action_scores.items()
                if value == optimal_value
            ]
        )
        return optimal_action

    def _batch_expert_rollout(
        self,
        source: List[List[int]],
        target: List[List[int]],
        alignment: torch.Tensor,
        prediction: List[List[int]],
        not_complete: torch.Tensor,
    ) -> List[List[int]]:
        """Performs expert rollout over batch."""
        return [
            (self._expert_rollout(s, t, align, pred) if nc else self.end_idx)
            for s, t, align, pred, nc in zip(
                source, target, alignment, prediction, not_complete
            )
        ]

    def _update_prediction(
        self,
        action: List[actions.Edit],
        source: List[int],
        alignment: torch.Tensor,
        prediction: List[List[str]],
    ) -> torch.Tensor:
        """Batch updates prediction and alignment information from actions.

        Args:
           action (List[actions.Edit]): valid actions, one per item in
                batch.
           source (List[int]): source strings, one per item in batch.
           alignment (torch.Tensor): index of current symbol for each item
                in batch.
           prediction (List[List[str]]): current predictions for each item
                in batch, one list of symbols per item.

        Return:
            torch.Tensor: new alignments for transduction.
        """
        alignment_update = torch.zeros_like(alignment)
        for i in range(len(source)):
            a = self.expert.actions.decode(action[i])
            if isinstance(a, actions.ConditionalCopy):
                symb = source[i][alignment[i]]
                prediction[i].append(symb)
                alignment_update[i] += 1
            elif isinstance(a, actions.ConditionalDel):
                alignment_update[i] += 1
            elif isinstance(a, actions.ConditionalIns):
                prediction[i].append(a.new)
            elif isinstance(a, actions.ConditionalSub):
                alignment_update[i] += 1
                prediction[i].append(a.new)
            elif isinstance(a, actions.End):
                prediction[i].append(special.END_IDX)
            else:
                raise expert.ActionError(f"Unknown action: {action[i]}")
        return alignment + alignment_update

    @staticmethod
    def _log_sum_softmax_loss(
        logits: torch.Tensor, optimal_actions: List[int]
    ) -> torch.Tensor:
        """Computes log loss.

        After:
            Riezler, S. Prescher, D., Kuhn, J. and Johnson, M. 2000.
            Lexicalized stochastic modeling of constraint-based grammars using
            log-linear measures and EM training. In Proceedings of the 38th
            Annual Meeting of the Association for Computational
            Linguistics, pages 480–487.
        """
        opt_act = [
            log[actions] for log, actions in zip(logits, optimal_actions)
        ]
        log_sum_exp_terms = torch.stack(
            [torch.logsumexp(act, dim=-1) for act in opt_act]
        )
        normalization_term = torch.logsumexp(logits, -1)
        return log_sum_exp_terms - normalization_term

    def _get_loss_func(
        self,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        # Prevents base construction of unused loss function; we don't need a
        # separate one because the forward pass computes loss as a side-effect.
        return None

    def training_step(self, batch: data.PaddedBatch, batch_idx: int) -> Dict:
        """Runs one step of training.

        This is called by the PL Trainer.

        Args:
            batch (data.PaddedBatch)
            batch_idx (int).

        Returns:
            torch.Tensor: loss.
        """
        # Forward pass produces loss by default.
        _, loss = self(batch)
        self.log(
            "train_loss",
            loss,
            batch_size=len(batch),
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch: data.PaddedBatch, batch_idx: int) -> Dict:
        predictions, loss = self(batch)
        # Evaluation requires prediction as a tensor.
        predictions = self._convert_predictions(predictions)
        # Gets a dict of all eval metrics for this batch.
        val_eval_items_dict = {
            evaluator.name: evaluator.evaluate(
                predictions,
                batch.target.padded,
                predictions_finalized=True,
            )
            for evaluator in self.evaluators
        }
        val_eval_items_dict.update({"val_loss": loss})
        return val_eval_items_dict

    def predict_step(
        self, batch: data.PaddedBatch, batch_idx: int
    ) -> torch.Tensor:
        predictions, _ = self(batch)
        # Evaluation requires prediction tensor.
        return self._convert_predictions(predictions)

    def _convert_predictions(
        self, predictions: List[List[int]]
    ) -> torch.Tensor:
        """Converts prediction values to tensor for evaluator compatibility."""
        # FIXME: the two steps below may be partially redundant.
        # TODO: Clean this up and make it more efficient.
        max_len = len(max(predictions, key=len))
        for i, prediction in enumerate(predictions):
            pad = [self.end_idx] * (max_len - len(prediction))
            prediction.extend(pad)
            predictions[i] = torch.tensor(prediction)
        predictions = torch.stack(predictions)
        # This turns all symbols after the first END into PAD so prediction
        # tensors match gold tensors.
        return util.pad_tensor_after_end(predictions)

    def on_train_epoch_start(self) -> None:
        """Scheduler for oracle."""
        self.expert.roll_in_schedule(self.current_epoch)

    @staticmethod
    def _sample(log_probs: torch.Tensor) -> torch.Tensor:
        """Samples an action from a log-probability distribution."""
        dist = torch.exp(log_probs)
        rand = numpy.random.rand()
        for action, prob in enumerate(dist):
            rand -= prob
            if rand <= 0:
                break
        return action

    # Interface to implement.

    def get_decoder(self):
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError


class TransducerGRUModel(TransducerRNNModel, rnn.GRUModel):
    """Transducer with GRU backend."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_decoder(self) -> modules.GRUDecoder:
        return modules.GRUDecoder(
            bidirectional=False,
            decoder_input_size=self.decoder_input_size,
            dropout=self.dropout,
            embeddings=self.embeddings,
            embedding_size=self.embedding_size,
            layers=self.decoder_layers,
            hidden_size=self.hidden_size,
            num_embeddings=self.vocab_size,
        )

    @property
    def name(self) -> str:
        return "transducer GRU"


class TransducerLSTMModel(TransducerRNNModel, rnn.LSTMModel):
    """Transducer with LSTM backend."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_decoder(self) -> modules.LSTMDecoder:
        return modules.LSTMDecoder(
            bidirectional=False,
            decoder_input_size=self.decoder_input_size,
            dropout=self.dropout,
            embeddings=self.embeddings,
            embedding_size=self.embedding_size,
            layers=self.decoder_layers,
            hidden_size=self.hidden_size,
            num_embeddings=self.vocab_size,
        )

    @property
    def name(self) -> str:
        return "transducer LSTM"
