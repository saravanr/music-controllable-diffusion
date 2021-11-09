import os.path
from typing import Any

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
        if input_shape[0] < 100:
            scale = 1
        else:
            scale = 100

        input_dim = (input_shape[0] * input_shape[1]) // scale
        self._net = nn.Sequential(
            nn.Linear(input_shape[0] * input_shape[1], input_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(input_dim // 2, input_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.LeakyReLU(),
            nn.Linear(input_dim // 4, input_dim // 8)
        )

        self._fc_mean = nn.Linear(input_dim // 8, z_dim)
        self._fc_var = nn.Linear(input_dim // 8, z_dim)

    def forward(self, x):
        if type(x) == list:
            x = x[0]
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

        if output_shape[0] < 100:
            scale = 1
        else:
            scale = 100

        output_dim = (output_shape[0] * output_shape[1]) // scale
        self._net = nn.Sequential(
            nn.Linear(z_dim, output_dim // 4),
            nn.LeakyReLU(),
            nn.Linear(output_dim // 4, output_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(output_dim // 2, output_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(output_dim // 2, output_dim * scale)
        )

    def forward(self, z):
        output = self._net(z)
        # TODO
        output = output.reshape((-1, 1, self._output_shape[0], self._output_shape[1]))
        return output

    @property
    def z_dim(self):
        return self._z_dim


class SimpleVae(BaseModel):
    def __init__(self, z_dim=64, input_shape=(10000, 8), alpha=1.0, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.writer = SummaryWriter()
        self._encoder = Encoder(z_dim, input_shape=input_shape)
        self._decoder = Decoder(z_dim, output_shape=input_shape)
        self._z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self._z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self._alpha = alpha
        self._model_prefix = "SimpleVae"

    def forward(self, x):
        mean, var = self._encoder.forward(x)
        relu = nn.ReLU()
        var = relu(var)
        z = SimpleVae.sample(mean, var)
        x_hat = self._decoder(z)
        return z, x_hat, mean, var

    def loss_function(self, x_hat, x, qm, qv):
        pm = self._z_prior_m
        pv = self._z_prior_v
        output = torch.sigmoid(x_hat)
        recon_loss = func.mse_loss(output, x, reduction='mean')
        kl = self._kl_normal(qm, qv, pm, pv)
        loss = recon_loss - self._alpha * kl
        logs = {
            # Detach before appending to reduce memory consumption
            "recon_loss": recon_loss,
            "kl_loss": kl,
            "loss": loss
        }
        return loss, logs

    def step(self, batch, batch_idx):
        if type(batch) == list:
            x = batch[0]
        else:
            x = batch
        z, x_hat, qm, qv = self.forward(x)
        loss, logs = self.loss_function(x_hat, x, qm, qv)
        return loss, logs

    def sample_output(self, epoch):
        try:
            sample_file_name = os.path.join(self._output_dir, f"{self._model_prefix}-{epoch}.midi")
            device = self.get_device()
            rand_z = torch.rand(self._decoder.z_dim).to(device)
            rand_z.to(device)
            logits = model._decoder(rand_z)
            output = torch.sigmoid(logits)
            sample = output.to("cpu").detach().numpy()
            if sample.shape[2] < 100:
                # TODO
                import matplotlib.pyplot as plt
                plt.imshow(sample.reshape((28, 28)))
                plt.show()
            else:
                print(f"Generating midi sample file://{sample_file_name}")
                save_decoder_output_as_midi(sample, sample_file_name)
        except Exception as _e:
            print(f"Hit exception during sample_output - {_e}")


if __name__ == "__main__":
    print(f"Training simple VAE")
    model = SimpleVae(
        z_dim=2,
        input_shape=(28, 28),
        use_mnist_dms=True,
        sample_output_step=10,
        batch_size=100000
    )
    print(f"Training --> {model}")
    for epoch in range(1, 51):
        model.fit(epoch)
        if epoch % 10 == 0:
            model.test()
            model.sample_output(epoch)
            model.save(epoch)
