import os.path

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from data.midi_data_module import MidiDataModule
from models.base.base_model import BaseModel
from torch import nn
from torch.nn import functional as func
from pytorch_lightning import Trainer
from torch.utils.tensorboard import SummaryWriter

class Encoder(nn.Module):
    """
    VAE Encoder takes the input and maps it to latent representation Z
    """

    def __init__(self, z_dim, input_shape):
        super().__init__()
        self._z_dim = z_dim
        flattened_input = (input_shape[0] * input_shape[1]) // 10
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
        flattened_output = (output_shape[0] * output_shape[1]) // 10
        self._net = nn.Sequential(
            nn.Linear(z_dim, flattened_output // 4),
            nn.LeakyReLU(),
            nn.Linear(flattened_output // 4, flattened_output // 2),
            nn.LeakyReLU(),
            nn.Linear(flattened_output // 2, flattened_output // 2),
            nn.LeakyReLU(),
            nn.Linear(flattened_output // 2, flattened_output * 10)
        )

    def forward(self, z):
        output = self._net(z)
        output = output.reshape((-1, self._output_shape[0], self._output_shape[1]))
        return output


class SimpleVae(BaseModel):
    def __init__(self, z_dim, input_shape):
        super().__init__()
        self.writer = SummaryWriter()
        self._encoder = Encoder(z_dim, input_shape=input_shape)
        self._decoder = Decoder(z_dim, output_shape=input_shape)
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)

    @staticmethod
    def sample(mean, log_var):
        # Set all small values to epsilon
        std = func.softplus(log_var) + 1e-8
        q = torch.distributions.Normal(mean, std)
        z = q.rsample()
        return z

    def from_pretrained(self, checkpoint_path, z_dim, input_shape):
        return self.load_from_checkpoint(checkpoint_path, z_dim, input_shape)

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
        element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
        kl = element_wise.sum(-1)
        return kl

    def _kl(self, mean, var):
        eps = 1e-8
        kl_loss = -0.5 * torch.sum(1 + torch.log(var + eps) - torch.square(mean) - var, axis=-1)
        kl_loss = torch.mean(kl_loss)
        return kl_loss

    def forward(self, x):
        mean, var = self._encoder.forward(x)
        relu = nn.ReLU()
        var = relu(var)
        z = SimpleVae.sample(mean, var)
        x_hat = self._decoder(z)
        return z, x_hat, mean, var

    def step(self, batch, batch_idx):
        x = batch
        z, x_hat, pm, pv = self.forward(x)

        recon_loss = func.mse_loss(x_hat, x, reduction='mean')
        kl = self._kl(pm, pv)
        loss = recon_loss + kl
        recon_loss_scalar = recon_loss.detach().item()
        kl_loss_scalar = kl.detach().item()
        loss_scalar = loss.detach().item()
        logs = {
            # Detach before appending to reduce memory consumption
            "recon_loss": recon_loss_scalar,
            "kl_loss": kl_loss_scalar,
            "loss": loss_scalar
        }
        self.writer.add_scalar("Loss/train/recon", recon_loss_scalar, self.current_epoch)
        self.writer.add_scalar("Loss/train/kl", kl_loss_scalar, self.current_epoch)
        self.writer.add_scalar("Loss/train/loss", loss_scalar, self.current_epoch)

        return loss, logs

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss, logs = self.step(batch, batch_idx)
        print(f"Training Reconstruction Loss={logs['recon_loss']} KL Loss={logs['kl_loss']} Total Loss={logs['loss']}")
        if self.global_step % 20000 == 0:
            print(f"Saving model @ {self.global_step}...")
            self.trainer.save_checkpoint(f"simplevae.chkpoint.{self.global_step}")
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._lr)


if __name__ == "__main__":
    print(f"Training simple VAE")
    # torch.autograd.set_detect_anomaly(True)
    z_dim = 256
    input_shape = (10000, 8)
    model = SimpleVae(z_dim=z_dim, input_shape=input_shape)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    dm = MidiDataModule(
        data_dir=os.path.expanduser("~/midi_processed/"),
        batch_size=256,
    )
    trainer = Trainer(auto_scale_batch_size="power",
                      gpus=1)
    trainer.fit(model, dm)
    trainer.save_checkpoint("simplevae.chkpoint")



