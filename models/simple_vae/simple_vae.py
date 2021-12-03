import os.path
from typing import Any

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.nn import functional as func
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import utils.midi_utils
from data.midi_data_module import MAX_MIDI_ENCODING_ROWS, MIDI_ENCODING_WIDTH
from models.base.base_model import BaseModel
from utils.cuda_utils import get_device
from utils.midi_utils import save_decoder_output_as_midi
import matplotlib.pyplot as plt

import wandb

wandb.init(project="music-controllable-diffusion-midi", entity="saravanr")


class ExtractLSTMOutput(nn.Module):
    def __init__(self, extract_out=True):
        super(ExtractLSTMOutput, self).__init__()
        self._extract_out = extract_out

    def forward(self, x):
        out, hidden = x
        if self._extract_out:
            return out
        else:
            return hidden


class Reshape1DTo2D(nn.Module):
    def __init__(self, output_shape):
        super(Reshape1DTo2D, self).__init__()
        self._output_shape = output_shape

    def forward(self, x):
        x = x.view((-1, self._output_shape[0], self._output_shape[1]))
        return x


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
            scale = 1

        seq_len = input_shape[0]
        seq_width = input_shape[1]
        input_dim = (seq_len * seq_width) // scale

        self._net = nn.Sequential(
            nn.GRU(input_size=seq_len, hidden_size=seq_len, num_layers=4, batch_first=True, bidirectional=True),
            ExtractLSTMOutput(),
            nn.Flatten(start_dim=1),
            nn.Linear(input_dim * 2, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, input_dim // 8),
        )

        self._fc_mean = nn.Sequential(
            nn.Linear(input_dim // 8, z_dim),
        )
        self._fc_log_var = nn.Sequential(
            nn.Linear(input_dim // 8, z_dim),
        )
        self._device = get_device()
        self._seq_len = seq_len
        self._seq_width = seq_width

    def forward(self, x):

        x = x.reshape((-1, self._seq_width, self._seq_len))
        h = self._net(x)
        mean = self._fc_mean(h)
        log_var = self._fc_log_var(h)
        return mean, log_var


class Decoder(nn.Module):
    """
    VAE Decoder takes the latent Z and maps it to output of shape of the input
    """

    def __init__(self, z_dim, output_shape, clam_output=False):
        super(Decoder, self).__init__()
        self._z_dim = z_dim
        self._output_shape = output_shape
        seq_len = output_shape[0]
        seq_width = output_shape[1]

        if output_shape[0] < 100:
            scale = 1
        else:
            scale = 1

        output_dim = (seq_len * seq_width) // scale
        self._net = nn.Sequential(
            nn.Linear(z_dim, output_dim * scale),
            Reshape1DTo2D((seq_width, seq_len)),
            nn.GRU(input_size=seq_len, hidden_size=seq_len, num_layers=1, batch_first=True),
            ExtractLSTMOutput(),
            nn.Tanh()
        )
        self._seq_len = seq_len
        self._seq_width = seq_width
        self._clamp_output = clam_output

    def forward(self, z):
        output = self._net(z)
        if self._clamp_output:
            # Clamp output in (0, 1) to prevent errors in BCE
            output = torch.clamp(output, 1e-8, 1 - 1e-8)
        else:
            output = output.reshape((-1, self._seq_len, self._seq_width))
        return output

    @property
    def z_dim(self):
        return self._z_dim


class SimpleVae(BaseModel):
    def __init__(self, z_dim=64, input_shape=(10000, 8), alpha=1.0, *args: Any, **kwargs: Any):
        super(SimpleVae, self).__init__(*args, **kwargs)
        self.writer = SummaryWriter()
        self._encoder = Encoder(z_dim, input_shape=input_shape)

        self._pitches_shape = (input_shape[0], 1)
        self._velocities_shape = (input_shape[0], 1)
        self._start_times_shape = (input_shape[0], 1)
        self._end_times_shape = (input_shape[0], 1)
        self._instruments_shape = (input_shape[0], 1)
        self._programs_shape = (input_shape[0], 1)

        self._pitches_decoder = Decoder(z_dim, output_shape=self._pitches_shape)
        self._velocity_decoder = Decoder(z_dim, output_shape=self._velocities_shape)
        self._start_times_decoder = Decoder(z_dim, output_shape=self._start_times_shape)
        self._end_times_decoder = Decoder(z_dim, output_shape=self._end_times_shape)
        self._instruments_decoder = Decoder(z_dim, output_shape=self._instruments_shape)
        self._program_decoder = Decoder(z_dim, output_shape=self._programs_shape)

        self._alpha = alpha
        self._model_prefix = "SimpleVaeMidi"
        self._z_dim = z_dim

    @staticmethod
    def from_pretrained(checkpoint_path, z_dim, input_shape):
        print(f"Loading from {checkpoint_path}...")
        _model = SimpleVae(z_dim=z_dim, input_shape=input_shape)
        _model.load_state_dict(torch.load(checkpoint_path))
        return _model

    def forward(self, x):
        # Renorm X to midi

        mean, log_var = self._encoder(x)
        z = SimpleVae.reparameterize(mean, log_var)

        x_pitches = self._pitches_decoder(z)
        x_velocity = self._velocity_decoder(z)
        x_start_times = self._start_times_decoder(z)
        x_end_times = self._end_times_decoder(z)
        x_instruments = self._instruments_decoder(z)
        x_programs = self._program_decoder(z)

        x_hat = torch.vstack(
            (x_pitches.T, x_velocity.T, x_instruments.T, x_programs.T, x_start_times.T, x_end_times.T)).T
        return z, x_hat, mean, log_var

    @staticmethod
    def compute_midi_recon_loss(x, x_hat):
        # Compute losses based on variable types.
        x_pitches_hat = x_hat.T[0]
        x_velocity_hat = x_hat.T[1]
        x_instruments_hat = x_hat.T[2]
        x_programs_hat = x_hat.T[3]
        x_start_times_hat = x_hat.T[4]
        x_end_times_hat = x_hat.T[5]

        x_pitches = x.T[0]
        x_velocity = x.T[1]
        x_instruments = x.T[2]
        x_programs = x.T[3]
        x_start_times = x.T[4]
        x_end_times = x.T[5]

        pitches_loss = func.cross_entropy(x_pitches_hat, x_pitches, reduction='mean')
        velocity_loss = func.cross_entropy(x_velocity_hat, x_velocity, reduction='mean')
        instruments_loss = func.cross_entropy(x_instruments_hat, x_instruments, reduction='mean')
        program_loss = func.cross_entropy(x_programs_hat, x_programs, reduction='mean')
        start_times_loss = func.mse_loss(x_start_times_hat, x_start_times, reduction='sum')
        end_times_loss = func.mse_loss(x_end_times_hat, x_end_times, reduction='sum')

        # TODO: How to ensure end times are > start times
        recon_loss = pitches_loss + velocity_loss + instruments_loss + program_loss + start_times_loss + end_times_loss
        return recon_loss

    def loss_function(self, x_hat, x, mu, q_log_var):
        if self._use_mnist_dms:
            recon_loss = func.binary_cross_entropy(x_hat, x.view(-1, 784), reduction='sum')
        else:
            recon_loss = SimpleVae.compute_midi_recon_loss(x, x_hat)
        kl = self._kl_simple(mu, q_log_var)

        x_i = torch.arctanh(x).T[2]
        x_i_hat = torch.arctanh(x_hat).T[2]

        instrument_loss = func.mse_loss(x_i_hat, x_i, reduction='mean')

        generated_unique_instruments_count = torch.unique((torch.arctanh(x_hat).T[2] * 254 + 127).to(torch.int)).size(
            dim=0)
        wandb.log({'instrument_count': float(generated_unique_instruments_count)})

        loss = recon_loss + self.alpha * kl
        return loss, kl, recon_loss, instrument_loss

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
                if self._use_mnist_dms:
                    # 16 for 4x4 set of numbers
                    rand_z = torch.randn(16, self._decoder.z_dim).to(device)
                    rand_z.to(device)
                    output = self._decoder(rand_z)
                    samples = output.to("cpu").detach().numpy()
                    samples = samples.reshape((-1, 28, 28))
                    SimpleVae.plot_image_grid(samples)
                else:
                    rand_z = torch.randn(self._pitches_decoder.z_dim).to(device)
                    rand_z.to(device)

                    x_pitches = self._pitches_decoder(rand_z)
                    x_velocity = self._velocity_decoder(rand_z)
                    x_start_times = self._start_times_decoder(rand_z)
                    x_end_times = self._end_times_decoder(rand_z)
                    x_instruments = self._instruments_decoder(rand_z)
                    x_programs = self._program_decoder(rand_z)
                    output = torch.vstack(
                        (x_pitches.T, x_velocity.T, x_instruments.T, x_programs.T, x_start_times.T, x_end_times.T)).T

                    sample = output.to("cpu").detach().numpy()
                    sample_file_name = os.path.join(self._output_dir,
                                                    f"{self._model_prefix}-{wandb.run.name}-{epoch}.midi")
                    print(f"Generating midi sample file://{sample_file_name}")
                    save_decoder_output_as_midi(sample, sample_file_name)
        except Exception as _e:
            print(f"Hit exception during sample_output - {_e}")

    @property
    def alpha(self):
        return self._alpha


if __name__ == "__main__":
    print(f"Training simple VAE")
    batch_size = 2048
    train_mnist = False
    _alpha = 1000
    if train_mnist:
        _z_dim = 20
        model = SimpleVae(
            alpha=_alpha,
            z_dim=_z_dim,
            input_shape=(28, 28),
            use_mnist_dms=True,
            sample_output_step=10,
            batch_size=batch_size
        )
    else:
        _z_dim = 1024
        model = SimpleVae(
            alpha=_alpha,
            z_dim=_z_dim,
            input_shape=(MAX_MIDI_ENCODING_ROWS, MIDI_ENCODING_WIDTH),
            use_mnist_dms=False,
            sample_output_step=10,
            batch_size=batch_size
        )
    print(f"Training --> {model}")

    max_epochs = 10000
    wandb.config = {
        "learning_rate": model.lr,
        "z_dim": _z_dim,
        "epochs": max_epochs,
        "batch_size": batch_size,
        "alpha": model.alpha
    }

    _optimizer = model.configure_optimizers()
    model.setup()
    for _epoch in range(1, max_epochs + 1):
        model.train()
        model.fit(_epoch, _optimizer)
        model.eval()
        model.test()
        if _epoch % 10 == 0:
            model.eval()
            model.sample_output(_epoch)
            model.sample_output(999999 + _epoch)
            model.save(_epoch)
