from pathlib import Path

import pandas as pd
import torch

from src.logger.utils import plot_spectrogram
from src.metrics.metrics import *
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker, batch_idx: int):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]

        if self.is_train and batch_idx % self.grad_accum_steps == 0:
            self.optimizer.zero_grad()

        with torch.autocast(device_type=self.device):
            outputs = self.model(**batch)
            batch.update(outputs)

            all_losses = self.criterion(**batch)
            batch.update(all_losses)

        if self.is_train:
            # sum of all losses is always called loss
            loss = batch["loss"] / self.grad_accum_steps
            self.scaler.scale(loss).backward()

            if batch_idx % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                self._clip_grad_norm()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        with torch.no_grad():
            for loss_name in self.config.writer.loss_names:
                metrics.update(loss_name, batch[loss_name])
                if self.writer is not None:
                    mode = "train" if self.is_train else "val"
                    self.writer.add_scalar(f"{mode}/{loss_name}", batch[loss_name])

            should_log_metrics = True
            if not self.is_train:
                n = self.config.trainer.log_inference_every_n_epochs
                if self.epochs % n != 0:
                    should_log_metrics = False

            if should_log_metrics:
                for met in metric_funcs:
                    metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch)
        else:
            # Log Stuff
            self.log_spectrogram(**batch)
            self.log_predictions(**batch)

    def log_spectrogram(self, s1_spectrogram, s2_spectrogram, mix_spectrogram, **batch):
        spectrogram_for_plot = s1_spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("s1_spectrogram", image)

        spectrogram_for_plot = s2_spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("s2_spectrogram", image)

        spectrogram_for_plot = mix_spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("mix_spectrogram", image)

    def log_predictions(
        self,
        s1_pred,
        s2_pred,
        s1_audio,
        s2_audio,
        mix_audio,
        mix_path,
        examples_to_log=5,
        **batch,
    ):
        tuples = list(zip(s1_pred, s2_pred, s1_audio, s2_audio, mix_audio, mix_path))
        si_snr = SI_SNR_Metric()
        snri = SI_SNRi_Metric(name="snri")
        pesq = PESQ_Metric()
        stoi = STOI_Metric()

        rows = {}
        for s1_p, s2_p, s1_a, s2_a, mix_a, mix_p in tuples[:examples_to_log]:
            self.writer.add_audio("mix_audio", mix_a, 16000)
            self.writer.add_audio("s1_groud_truth", s1_a, 16000)
            self.writer.add_audio("s2_groud_truth", s2_a, 16000)
            self.writer.add_audio("s1_separated", s1_p, 16000)
            self.writer.add_audio("s2_separated", s2_p, 16000)
            rows[Path(mix_p).name] = {
                "si_snr_speaker_1": si_snr.calc_metric(preds=s1_p, targets=s1_a),
                "si_snr_speaker_2": si_snr.calc_metric(preds=s2_p, targets=s2_a),
                "snri": snri.forward(mix_a, s1_p, s2_p, s1_a, s2_a),
                # "pesq_speaker_1": pesq.calc_metric(preds=s1_p, targets=s1_a),
                # "pesq_speaker_2": pesq.calc_metric(preds=s2_p, targets=s2_a),
                # "stoi_speaker_1": stoi.calc_metric(preds=s1_p, targets=s1_a),
                # "stoi_speaker_2": stoi.calc_metric(preds=s2_p, targets=s2_a),
            }
        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )
