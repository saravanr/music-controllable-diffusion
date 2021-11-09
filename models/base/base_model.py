from typing import Any

import torch
import os
from datetime import datetime
from pytorch_lightning import LightningModule, Trainer
from data.midi_data_module import MidiDataModule
import torch.nn.functional as F


class BaseModel(LightningModule):
    """
    Base model class that will be inherited by all model types
    """

    def __init__(self, lr=1e-4,
                 data_dir=os.path.expanduser("~/midi_processed/"),
                 output_dir=os.path.expanduser("~/model-archive/"),
                 num_gpus=1,
                 sample_output_step=200,
                 save_checkpoint_every=1000,
                 emit_tensorboard_scalars=True,
                 *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        now = datetime.now()
        now_str = now.strftime(format="%d-%h-%y-%s")
        self._data_dir = data_dir
        self._output_dir = os.path.join(output_dir, now_str)
        os.makedirs(self._output_dir)
        self._lr = lr
        self._dms = MidiDataModule(data_dir)
        self._model_prefix = "base-model"
        self._num_gpus = num_gpus
        self._sample_output_step = sample_output_step
        self._save_checkpoint_every = save_checkpoint_every
        self._emit_tensorboard_scalars = emit_tensorboard_scalars

    @staticmethod
    def sample(mean, log_var):
        # Set all small values to epsilon
        std = F.softplus(log_var) + 1e-8
        q = torch.distributions.Normal(mean, std)
        z = q.rsample()
        return z

    def from_pretrained(self, checkpoint_path):
        print(f"Loading from {checkpoint_path}...")
        return self.load_from_checkpoint(checkpoint_path)

    @staticmethod
    def _kl_normal(qm, qv, pm, pv):
        """
        Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
        sum over the last dimension

        Args:
            qm: tensor: (batch, dim): q mean
            qv: tensor: (batch, dim): q variance
            pm: tensor: (batch, dim): p mean
            pv: tensor: (batch, dim): p variance

        Return:
            kl: tensor: (batch,): kl between each sample
        """
        eps = 1e-8
        qv = qv + eps
        element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
        kl = element_wise.sum(-1)
        kl = torch.mean(kl)
        return kl

    @staticmethod
    def _kl(mean, var):
        eps = 1e-8
        kl_loss = -0.5 * torch.sum(1 + torch.log(var + eps) - torch.square(mean) - var, axis=-1)
        kl_loss = torch.mean(kl_loss)
        return kl_loss

    def enable_debugging(self):
        torch.autograd.set_detect_anomaly(True)

    def get_device(self):
        if self._num_gpus > 0:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            return 'cpu'

    def fit(self):
        device = self.get_device()
        self.to(device)

        trainer = Trainer(auto_scale_batch_size="power", gpus=self._num_gpus)
        trainer.fit(self, self._dms)
        trainer.save_checkpoint(os.path.join(self._output_dir, f"{self._model_prefix}-final.checkpoint"))
