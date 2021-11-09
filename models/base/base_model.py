from typing import Any

import torch
import os
from datetime import datetime
from pl_bolts.datamodules import MNISTDataModule
from data.midi_data_module import MidiDataModule
import torch.nn.functional as F
import torch.optim as optim


class BaseModel(torch.nn.Module):
    """
    Base model class that will be inherited by all model types
    """

    def __init__(self, lr=1e-4,
                 data_dir=os.path.expanduser("~/midi_processed/"),
                 output_dir=os.path.expanduser("~/model-archive/"),
                 num_gpus=1,
                 batch_size=30000,
                 sample_output_step=200,
                 save_checkpoint_every=1000,
                 emit_tensorboard_scalars=True,
                 use_mnist_dms=False,
                 *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        now = datetime.now()
        now_str = now.strftime(format="%d-%h-%y-%s")
        self._data_dir = data_dir
        self._output_dir = os.path.join(output_dir, now_str)
        os.makedirs(self._output_dir)
        self._lr = lr
        if use_mnist_dms:
            self._dms = MNISTDataModule(num_workers=12,
                                        pin_memory=True)
        else:
            self._dms = MidiDataModule(data_dir)
        self._model_prefix = "base-model"
        self._num_gpus = num_gpus
        self._sample_output_step = sample_output_step
        self._save_checkpoint_every = save_checkpoint_every
        self._emit_tensorboard_scalars = emit_tensorboard_scalars
        self.batch_size = batch_size
        self._dms.setup()

    @staticmethod
    def sample(mean, log_var):
        # Set all small values to epsilon
        std = F.softplus(log_var) + 1e-8
        q = torch.distributions.Normal(mean, std)
        z = q.rsample()
        return z

    def from_pretrained(self, checkpoint_path):
        print(f"Loading from {checkpoint_path}...")
        model = self.__class__()
        model.load_state_dict(torch.load(checkpoint_path))
        return model

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
            return device
        else:
            return 'cpu'

    def loss_function(self, x_hat, x, qm, qv):
        raise NotImplementedError(f"Please implement loss_function()")

    def step(self, batch, batch_idx):
        raise NotImplementedError(f"Please implement step()")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._lr)

    def sample_output(self, epoch):
        raise NotImplementedError(f"Please implement sample_output()")

    def fit(self, epoch):
        device = self.get_device()
        self.to(device)
        optimizer = self.configure_optimizers()

        self.train()
        train_loss = 0

        for batch_idx, (batch, _) in enumerate(self._dms.train_dataloader()):
            batch = batch.cuda()
            optimizer.zero_grad()
            loss, logs = self.step(batch, batch_idx)
            loss.backward()
            batch_loss = loss.detach().item()
            train_loss += batch_loss
            optimizer.step()

        print(f'====> Train Loss = {train_loss / len(self._dms.train_dataloader())} Epoch = {epoch}')

    def save(self, epoch):
        model_save_path = os.path.join(self._output_dir, f"{self._model_prefix}-epoch-{epoch}.checkpoint")
        torch.save(self.state_dict(), model_save_path)

    def test(self):
        self.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_idx, (batch, _) in enumerate(self._dms.test_dataloader()):
                batch = batch.cuda()
                loss, logs = self.step(batch, batch_idx)
                batch_loss = loss.detach().item()
                test_loss += batch_loss

        test_loss /= len(self._dms.test_dataloader())
        print(f"====>  Test Loss = {test_loss}")

