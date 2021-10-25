import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT

from models.base.base_model import BaseModel
from torch import nn
from torch.nn import functional as func


class Encoder(nn.Module):
    """
    VAE Encoder takes the input and maps it to latent representation Z
    """

    def __init__(self, z_dim, input_shape):
        super().__init__()
        self._z_dim = z_dim
        flattened_input = input_shape[0] * input_shape[1]

        self._net = nn.Sequential(
            nn.Linear(input_shape[0] * input_shape[1], flattened_input / 2),
            nn.LeakyReLU(),
            nn.Linear(flattened_input / 2, flattened_input / 2),
            nn.LeakyReLU(),
            nn.Linear(flattened_input / 2, flattened_input / 4),
            nn.LeakyReLU(),
            nn.Linear(flattened_input / 4, flattened_input / 8)
        )

        self._fc_mean = nn.Linear(flattened_input/8, z_dim)
        self._fc_var = nn.Linear(flattened_input/8, z_dim)

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
        flattened_output = output_shape[0] * output_shape[1]
        self._net = nn.Sequential(
            nn.Linear(z_dim, flattened_output / 4),
            nn.LeakyReLU(),
            nn.Linear(flattened_output / 4, flattened_output / 2),
            nn.LeakyReLU(),
            nn.Linear(flattened_output / 2, flattened_output / 2),
            nn.LeakyReLU(),
            nn.Linear(flattened_output / 2, flattened_output)
        )

    def forward(self, z):
        output = self._net(z)
        return output


class SimpleVae(BaseModel):
    def __init__(self, z_dim, input_shape):
        super().__init__()
        self._encoder = Encoder(z_dim, input_shape=input_shape)
        self._decoder = Decoder(z_dim, output_shape=input_shape)

    @staticmethod
    def sample(mean, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(std))
        q = torch.distributions.Normal(mean, std)
        z = q.rsample()
        return z

    @staticmethod
    def _kl(mean, log_var):
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mean ** 2 - log_var.exp(), dim =1), dim=0)
        return kl_loss

    def forward(self, x):
        mean, log_var = self._encoder.forward(x)
        z = SimpleVae.sample(mean, log_var)
        x_hat = self._decoder(z)
        return z, x_hat, mean, log_var

    def step(self, batch, batch_idx):
        x = batch
        z, x_hat, pm, pv = self.forward(x)

        recon_loss = func.mse_loss(x_hat, x, reduction='mean')
        kl = SimpleVae._kl(pm, pv)
        loss = recon_loss + kl
        logs = {
            "recon_loss": recon_loss,
            "kl_loss": kl,
            "loss": loss
        }
        return loss, logs

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss, logs = self.step(batch, batch_idx)
        print(f"Training Reconstruction Loss={logs['recon_loss']} KL Loss={logs['kl_loss']} Total Loss={logs['loss']}")
        return loss





