"""Initialize a Pytorch model wrapper that feeds into Model Runner."""

import warnings

import torch
from torch.utils.tensorboard import SummaryWriter

from easy_tpp.utils import RunnerPhase, set_optimizer, set_device


class TorchModelWrapper:
    def __init__(self, model, base_config, model_config, trainer_config):
        """A wrapper class for Torch backends.

        Args:
            model (BaseModel): a TPP model.
            base_config (EasyTPP.Config): basic configs.
            model_config (EasyTPP.ModelConfig): model spec configs.
            trainer_config (EasyTPP.TrainerConfig): trainer spec configs.
        """
        self.model = model
        self.base_config = base_config
        self.model_config = model_config
        self.trainer_config = trainer_config

        self.model_id = self.base_config.model_id

        # Sometimes PyTorch may not switch the active device context for all operations.
        # This can cause illegal memory access errors.
        if self.trainer_config.gpu != -1:
            torch.cuda.set_device(self.trainer_config.gpu)

        self.device = set_device(self.trainer_config.gpu)
        self.model.to(self.device)

        if self.model_config.is_training:
            optimizer = self.trainer_config.optimizer
            self.learning_rate = self.trainer_config.learning_rate
            self.opt = set_optimizer(
                optimizer,
                self.model.parameters(),
                self.learning_rate,
            )

        self.train_summary_writer, self.valid_summary_writer = None, None
        if self.trainer_config.use_tfb:
            self.train_summary_writer = SummaryWriter(
                log_dir=self.base_config.specs["tfb_train_dir"]
            )
            self.valid_summary_writer = SummaryWriter(
                log_dir=self.base_config.specs["tfb_valid_dir"]
            )

    def restore(self, ckpt_dir):
        """Load the checkpoint to restore the model.

        Args:
            ckpt_dir (str): path for the checkpoint.
        """
        state_dict = torch.load(ckpt_dir, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)

    def save(self, ckpt_dir):
        """Save the checkpoint for the model.

        Args:
            ckpt_dir (str): path for the checkpoint.
        """
        torch.save(self.model.state_dict(), ckpt_dir)

    def write_summary(self, epoch, kv_pairs, phase):
        """Write key-value metrics into tensorboard.

        Args:
            epoch (int): epoch index.
            kv_pairs (dict): metrics dict.
            phase (RunnerPhase): runner phase.
        """
        if not self.trainer_config.use_tfb:
            return

        summary_writer = None
        if phase == RunnerPhase.TRAIN:
            summary_writer = self.train_summary_writer
        elif phase == RunnerPhase.VALIDATE:
            summary_writer = self.valid_summary_writer

        if summary_writer is None:
            return

        for k, v in kv_pairs.items():
            if k != "num_events":
                summary_writer.add_scalar(k, v, epoch)

        summary_writer.flush()

    def close_summary(self):
        """Close tensorboard summary writers."""
        if self.train_summary_writer is not None:
            self.train_summary_writer.close()

        if self.valid_summary_writer is not None:
            self.valid_summary_writer.close()

    @staticmethod
    def _to_int_num_event(num_event):
        """Convert num_event to Python int safely."""
        if torch.is_tensor(num_event):
            return int(num_event.detach().cpu().item())
        return int(num_event)

    def _predict_type_at_ground_truth_time(self, batch):
        """Predict next event type at the ground-truth next event time.

        This metric is different from normal acc:
            normal acc:
                first predict next time by thinning, then predict type at that predicted time.

            acc_gt_time:
                use ground-truth next time, only evaluate type prediction.

        Args:
            batch: list returned by BatchEncoding.values():
                [time_seqs, time_delta_seqs, type_seqs, seq_non_pad_mask, attention_mask]

        Returns:
            torch.LongTensor: [batch_size, seq_len - 1]
        """
        time_seqs, time_delta_seqs, type_seqs, _, attention_mask = batch

        # Prefix events: [t_0, ..., t_{N-1}]
        prefix_time = time_seqs[:, :-1]
        prefix_dtime = time_delta_seqs[:, :-1]
        prefix_type = type_seqs[:, :-1]

        # Ground-truth next inter-event time: [dt_1, ..., dt_N]
        gt_next_dtime = time_delta_seqs[:, 1:].unsqueeze(-1)

        kwargs = {"compute_last_step_only": False}

        # Attention-based models should receive the correct prefix mask.
        # Non-attention models accept **kwargs and ignore it.
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask[:, :-1, :-1]

        # Important:
        # Most EasyTPP models expect sample_dtimes as inter-event times.
        # AttNHP's implementation expects absolute sample times in its loss path.
        if self.model_id == "AttNHP":
            sample_times = time_seqs[:, 1:].unsqueeze(-1)
        else:
            sample_times = gt_next_dtime

        lambdas = self.model.compute_intensities_at_sample_times(
            prefix_time,
            prefix_dtime,
            prefix_type,
            sample_times,
            **kwargs,
        )

        # Expected shape: [B, L-1, 1, num_event_types]
        # Be tolerant to implementations that may return [B, L-1, num_event_types].
        if lambdas.dim() == 4:
            lambdas = lambdas.squeeze(-2)

        pred_type_gt_time = torch.argmax(lambdas, dim=-1)
        return pred_type_gt_time

    def _validation_predictions(self, batch):
        """Generate validation predictions.

        Returns:
            pred_dtime: np.ndarray or None, [B, L-1]
            pred_type: np.ndarray or None, [B, L-1]
            pred_type_gt_time: np.ndarray or None, [B, L-1]
            label_dtime: np.ndarray, [B, L-1]
            label_type: np.ndarray, [B, L-1]
            mask: np.ndarray or None, [B, L-1]
        """
        label_dtime, label_type, mask = None, None, None
        pred_dtime, pred_type, pred_type_gt_time = None, None, None

        if batch[1] is not None and batch[2] is not None:
            label_dtime = batch[1][:, 1:].detach().cpu().numpy()
            label_type = batch[2][:, 1:].detach().cpu().numpy()

        if batch[3] is not None:
            mask = batch[3][:, 1:].detach().cpu().numpy()

        # 1) Normal EasyTPP one-step prediction:
        #    predict next time by thinning, then predict type at predicted time.
        if getattr(self.model, "event_sampler", None) is not None:
            pred_dtime_t, pred_type_t = self.model.predict_one_step_at_every_event(
                batch=batch
            )
            pred_dtime = pred_dtime_t.detach().cpu().numpy()
            pred_type = pred_type_t.detach().cpu().numpy()
        else:
            warnings.warn(
                f"{self.model_id} has no event_sampler. "
                "Normal acc/rmse predictions will be None. "
                "acc_gt_time can still be computed if compute_intensities_at_sample_times is implemented.",
                RuntimeWarning,
            )

        # 2) New prediction for acc_gt_time:
        #    predict type at ground-truth next event time.
        if hasattr(self.model, "predict_type_at_ground_truth_time"):
            pred_type_gt_time_t = self.model.predict_type_at_ground_truth_time(
                batch=batch
            )
        else:
            pred_type_gt_time_t = self._predict_type_at_ground_truth_time(batch)

        pred_type_gt_time = pred_type_gt_time_t.detach().cpu().numpy()

        return pred_dtime, pred_type, pred_type_gt_time, label_dtime, label_type, mask

    def run_batch(self, batch, phase):
        """Run one batch.

        Args:
            batch (EasyTPP.BatchEncoding): preprocessed batch data.
            phase (RunnerPhase): train / validate / predict.

        Returns:
            For training / validation:
                loss, num_event, predictions, labels, mask

            For prediction:
                predictions, labels
        """
        batch = batch.to(self.device).values()

        if phase in (RunnerPhase.TRAIN, RunnerPhase.VALIDATE):
            is_training = phase == RunnerPhase.TRAIN
            self.model.train(is_training)

            # FullyNN needs grad even in validation stage.
            grad_flag = is_training if self.model_id != "FullyNN" else True

            with torch.set_grad_enabled(grad_flag):
                loss, num_event = self.model.loglike_loss(batch)

            num_event_int = self._to_int_num_event(num_event)

            # Default values. This prevents UnboundLocalError in train phase.
            pred_dtime = None
            pred_type = None
            pred_type_gt_time = None
            label_dtime = None
            label_type = None
            mask = None

            if is_training:
                if num_event_int <= 0:
                    raise RuntimeError(
                        "num_event is zero in training batch. "
                        "Please check seq_non_pad_mask, pad_token_id, and input sequence lengths."
                    )

                self.opt.zero_grad()
                (loss / num_event).backward()
                self.opt.step()

            else:
                self.model.eval()
                with torch.no_grad():
                    (
                        pred_dtime,
                        pred_type,
                        pred_type_gt_time,
                        label_dtime,
                        label_type,
                        mask,
                    ) = self._validation_predictions(batch)

            predictions = (pred_dtime, pred_type, pred_type_gt_time)
            labels = (label_dtime, label_type)

            return loss.item(), num_event_int, predictions, labels, (mask,)

        # PREDICT / generation phase
        self.model.eval()
        with torch.no_grad():
            pred_dtime, pred_type, label_dtime, label_type = (
                self.model.predict_multi_step_since_last_event(batch=batch)
            )

        pred_dtime = pred_dtime.detach().cpu().numpy()
        pred_type = pred_type.detach().cpu().numpy()
        label_dtime = label_dtime.detach().cpu().numpy()
        label_type = label_type.detach().cpu().numpy()

        return (pred_dtime, pred_type), (label_dtime, label_type)
