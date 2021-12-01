import random
from random import shuffle

import PIL
import torch
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from tts.base import BaseTrainer
from tts.logger.utils import plot_spectrogram_to_buf
from tts.utils import inf_loop, MetricTracker
from tts.model import Vocoder
from tts.model import GraphemeAligner
from tts.featurizer import MelSpectrogram, MelSpectrogramConfig


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion_mel,
            criterion_dur,
            optimizer,
            config,
            device,
            data_loader,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, optimizer, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.data_loader = data_loader
        self.criterion_mel = criterion_mel
        self.criterion_dur = criterion_dur
        self.vocoder = Vocoder().to(device)
        self.galigner = GraphemeAligner().to(device)
        self.featurizer = MelSpectrogram(MelSpectrogramConfig()).to(device)

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.lr_scheduler = lr_scheduler
        self.log_step = 50

        self.train_metrics = MetricTracker(
            "mel_loss", "dur_loss", "grad norm", writer=self.writer
        )

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.data_loader, desc="train", total=self.len_epoch)
        ):
            batch = batch.to(self.device)
            batch.melspec = self.featurizer(batch.waveform)

            batch.melspec_length = batch.melspec.shape[-1] - batch.melspec.eq(-11.5129251)[:, 0, :].sum(dim=-1)
            batch.melspec_length = batch.melspec_length

            with torch.no_grad():
                durations = self.galigner(
                    batch.waveform, batch.waveform_length, batch.transcript
                ).to(self.device)

                batch.durations = durations * batch.melspec_length.unsqueeze(-1)
            try:
                ml, dl = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                # print(batch)
                print(e)
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} MELLoss: {:.6f}, DURLoss: {:.6f}".format(
                        epoch, self._progress(batch_idx), ml, dl
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                self._log_scalars(self.train_metrics)
            if batch_idx >= self.len_epoch:
                break
        log = self.train_metrics.result()

        self._check_examples()

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = batch.to(self.device)
        self.optimizer.zero_grad()
        outputs, dur_preds = self.model(batch.tokens, batch.durations.int())

        batch.melspec_prediction = outputs
        batch.durations_prediction = dur_preds
        batch.durations = batch.durations.float()

        mel_loss = self.criterion_mel(batch)
        dur_loss = self.criterion_dur(batch)

        mel_loss.backward(retain_graph=True)
        dur_loss.backward()
        self._clip_grad_norm()
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        metrics.update("mel_loss", mel_loss.item())
        metrics.update("dur_loss", dur_loss.item())

        return mel_loss.item(), dur_loss.item()

    def _check_examples(self):
        # TODO: implement right valid epoch
        self.model.eval()
        with torch.no_grad():
            for batch in self.data_loader:
                batch.to(self.device)
                output = self.model(batch.tokens, None)
                break
            prediction_wav = self.vocoder.inference(output[0].unsqueeze(0).transpose(-1, -2)).cpu()
            self._log_spectrogram(output[:1, :, :])
            self._log_audio("pred_wav", prediction_wav)
            self._log_audio("true_wav", batch.waveform[0])

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch)
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram.cpu().log()))
        self.writer.add_image("spectrogram", ToTensor()(image))

    def _log_audio(self, audio_name, wav):
        self.writer.add_audio(audio_name, wav, sample_rate=22050)

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
