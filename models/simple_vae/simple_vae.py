import os.path

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.nn import functional as func
from torch.utils.tensorboard import SummaryWriter

from models.base.base_model import BaseModel
from utils.midi_utils import save_decoder_output_as_midi


class Encoder(nn.Module):
    """
    VAE Encoder takes the input and maps it to latent representation Z
    """

    def __init__(self, z_dim, input_shape):
        super().__init__()
        self._z_dim = z_dim
        flattened_input = (input_shape[0] * input_shape[1]) // 100
        self._net = nn.Sequential(
            nn.Linear(input_shape[0] * input_shape[1], flattened_input // 2),
            nn.LeakyReLU(),
            nn.Linear(flattened_input // 2, flattened_input // 2),
            nn.LeakyReLU(),
            nn.Linear(flattened_input // 2, flattened_input // 4),
            nn.LeakyReLU(),
            nn.Linear(flattened_input // 4, flattened_input // 8)
        )

        self._fc_mean = nn.Linear(flattened_input // 8, z_dim)
        self._fc_var = nn.Linear(flattened_input // 8, z_dim)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        h = self._net(x)
        mean = self._fc_mean(h)
        var = self._fc_var(h)
        return mean, var


class Decoder(nn.Module):
    """
    VAE Decoder takes the latent Z and maps it to output of shape of the input
    """

    def __init__(self, z_dim, output_shape):
        super().__init__()
        self._z_dim = z_dim
        self._output_shape = output_shape
        flattened_output = (output_shape[0] * output_shape[1]) // 100
        self._net = nn.Sequential(
            nn.Linear(z_dim, flattened_output // 4),
            nn.LeakyReLU(),
            nn.Linear(flattened_output // 4, flattened_output // 2),
            nn.LeakyReLU(),
            nn.Linear(flattened_output // 2, flattened_output // 2),
            nn.LeakyReLU(),
            nn.Linear(flattened_output // 2, flattened_output * 100)
        )

    def forward(self, z):
        output = self._net(z)
        output = output.reshape((-1, self._output_shape[0], self._output_shape[1]))
        return output

    @property
    def z_dim(self):
        return self._z_dim


class SimpleVae(BaseModel):
    def __init__(self, z_dim=64, input_shape=(10000, 8)):
        super().__init__()
        self.writer = SummaryWriter()
        self._encoder = Encoder(z_dim, input_shape=input_shape)
        self._decoder = Decoder(z_dim, output_shape=input_shape)
        self._z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self._z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self._alpha = 3
        self._model_prefix = "SimpleVae"

    def forward(self, x):
        mean, var = self._encoder.forward(x)
        relu = nn.ReLU()
        var = relu(var)
        z = SimpleVae.sample(mean, var)
        x_hat = self._decoder(z)
        return z, x_hat, mean, var

    def step(self, batch, batch_idx):
        x = batch
        z, x_hat, qm, qv = self.forward(x)
        pm = self._z_prior_m
        pv = self._z_prior_v
        recon_loss = func.mse_loss(x_hat, x, reduction='mean')
        kl = self._kl_normal(qm, qv, pm, pv)
        loss = recon_loss - self._alpha * kl
        recon_loss_scalar = recon_loss.detach().item()
        kl_loss_scalar = kl.detach().item()
        loss_scalar = loss.detach().item()
        logs = {
            # Detach before appending to reduce memory consumption
            "recon_loss": recon_loss_scalar,
            "kl_loss": kl_loss_scalar,
            "loss": loss_scalar
        }

        if self._emit_tensorboard_scalars:
            self.writer.add_scalar("Loss/train/recon", recon_loss_scalar, self.current_epoch)
            self.writer.add_scalar("Loss/train/kl", kl_loss_scalar, self.current_epoch)
            self.writer.add_scalar("Loss/train/loss", loss_scalar, self.current_epoch)

        return loss, logs

    def sample_output(self):
        try:
            sample_file_name = os.path.join(self._output_dir, f"{self._model_prefix}-{self.global_step}.midi")
            device = self.get_device()
            rand_z = torch.rand(self._decoder.z_dim).to(device)
            logits = model._decoder(rand_z)
            sample = logits.to("cpu").detach().numpy()
            print(f"Generating midi sample file://{sample_file_name}")
            save_decoder_output_as_midi(sample, sample_file_name)
        except Exception as _e:
            print(f"Hit exception - {_e}")

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss, logs = self.step(batch, batch_idx)
        print(f"Training Reconstruction Loss={logs['recon_loss']} KL Loss={logs['kl_loss']} Total Loss={logs['loss']}")
        if self.global_step > 0 and self.global_step % self._save_checkpoint_every == 0:
            print(f"Saving model @ {self.global_step}...")
            model_path = os.path.join(self._output_dir, f"{self._model_prefix}-{self.global_step}.checkpoint")
            self.trainer.save_checkpoint(model_path)
        if self.global_step % self._sample_output_step == 0:
            with torch.no_grad():
                self.sample_output()
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._lr)


if __name__ == "__main__":
    print(f"Training simple VAE")
    model = SimpleVae()
    model.fit()
