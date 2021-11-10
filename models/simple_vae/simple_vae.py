import os.path
from typing import Any

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.nn import functional as func
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from models.base.base_model import BaseModel
from utils.cuda_utils import get_device
from utils.midi_utils import save_decoder_output_as_midi
import matplotlib.pyplot as plt

import wandb

wandb.init(project="music-controllable-diffusion", entity="saravanr")


class Encoder(nn.Module):
    """
    VAE Encoder takes the input and maps it to latent representation Z
    """

    def __init__(self, z_dim, input_shape):
        super(Encoder, self).__init__()
        self._z_dim = z_dim
        if input_shape[0] < 100:
            scale = 1
        else:
            scale = 100

        input_dim = (input_shape[0] * input_shape[1]) // scale
        self._net = nn.Sequential(
            nn.Linear(input_shape[0] * input_shape[1], input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, input_dim // 8)
        )

        self._fc_mean = nn.Sequential(
            nn.Linear(input_dim // 8, z_dim),
        )
        self._fc_log_var = nn.Sequential(
            nn.Linear(input_dim // 8, z_dim),
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        h = self._net(x)

        mean = self._fc_mean(h)
        log_var = self._fc_log_var(h)
        return mean, log_var


class Decoder(nn.Module):
    """
    VAE Decoder takes the latent Z and maps it to output of shape of the input
    """

    def __init__(self, z_dim, output_shape):
        super(Decoder, self).__init__()
        self._z_dim = z_dim
        self._output_shape = output_shape

        if output_shape[0] < 100:
            scale = 1
        else:
            scale = 100

        output_dim = (output_shape[0] * output_shape[1]) // scale
        self._net = nn.Sequential(
            nn.Linear(z_dim, output_dim // 4),
            nn.ReLU(),
            nn.Linear(output_dim // 4, output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, output_dim * scale),
            nn.Sigmoid()
        )

    def forward(self, z):
        output = self._net(z)
        # Clamp output in (0, 1) to prevent errors in BCE
        output = torch.clamp(output, 1e-8, 1 - 1e-8)
        return output

    @property
    def z_dim(self):
        return self._z_dim


class SimpleVae(BaseModel):
    def __init__(self, z_dim=64, input_shape=(10000, 8), alpha=1.0, *args: Any, **kwargs: Any):
        super(SimpleVae, self).__init__(*args, **kwargs)
        self.writer = SummaryWriter()
        self._encoder = Encoder(z_dim, input_shape=input_shape)
        self._decoder = Decoder(z_dim, output_shape=input_shape)
        self._alpha = alpha
        self._model_prefix = "SimpleVae"

    def forward(self, x):
        mean, log_var = self._encoder(x)
        z = SimpleVae.reparameterize(mean, log_var)
        x_hat = self._decoder(z)
        return z, x_hat, mean, log_var

    def loss_function(self, x_hat, x, mu, q_log_var):
        recon_loss = func.binary_cross_entropy(x_hat, x.view(-1, 784), reduction='sum')
        kl = self._kl_simple(mu, q_log_var)
        loss = recon_loss + self.alpha * kl
        return loss

    def step(self, batch, batch_idx):
        x = batch
        z, x_hat, mu, q_log_var = self(x)
        loss = self.loss_function(x_hat, x, mu, q_log_var)
        return loss

    @staticmethod
    def plot_image_grid(samples):
        from mpl_toolkits.axes_grid1 import ImageGrid
        fig = plt.figure(figsize=(4., 4.))
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(4, 4),
                         axes_pad=0.1,
                         )

        for ax, im in zip(grid, samples):
            ax.imshow(im)

        plt.show()

    def sample_output(self, epoch):
        try:
            with torch.no_grad():
                device = get_device()
                if True:
                    #TODO
                    rand_z = torch.randn(16, self._decoder.z_dim).to(device)
                    rand_z.to(device)
                    output = self._decoder(rand_z)
                    samples = output.to("cpu").detach().numpy()
                    samples = samples.reshape((-1, 28, 28))
                    SimpleVae.plot_image_grid(samples)
                else:
                    rand_z = torch.randn(self._decoder.z_dim).to(device)
                    rand_z.to(device)
                    output = self._decoder(rand_z)
                    sample = output.to("cpu").detach().numpy()
                    sample_file_name = os.path.join(self._output_dir, f"{self._model_prefix}-{epoch}.midi")
                    print(f"Generating midi sample file://{sample_file_name}")
                    save_decoder_output_as_midi(sample, sample_file_name)
        except Exception as _e:
            print(f"Hit exception during sample_output - {_e}")

    @property
    def alpha(self):
        return self._alpha


if __name__ == "__main__":
    print(f"Training simple VAE")
    batch_size = 100
    model = SimpleVae(
        alpha=1,
        z_dim=20,
        input_shape=(28, 28),
        use_mnist_dms=True,
        sample_output_step=10,
        batch_size=batch_size
    )
    print(f"Training --> {model}")

    max_epochs = 10000
    wandb.config = {
        "learning_rate" : model.lr,
        "epochs":  max_epochs,
        "batch_size": batch_size,
        "alpha": model.alpha
    }

    _optimizer = model.configure_optimizers()
    for epoch in range(1, max_epochs + 1):
        model.fit(epoch, _optimizer)
        if epoch % 100 == 0:
            model.test()
            model.sample_output(epoch)
            model.save(epoch)
