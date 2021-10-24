import torch
import numpy as np
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
            nn.Linear(input_shape[0]*input_shape[1], flattened_input/2),
            nn.LeakyReLU(),
            nn.Linear(flattened_input/2, flattened_input/2),
            nn.LeakyReLU(),
            nn.Linear(flattened_input/2, flattened_input/4),
            nn.LeakyReLU(),
            nn.Linear(flattened_input/4, 2 * z_dim),
        )

    def forward(self, x):
        x = torch.flatten(x , start_dim=1)
        h = self._net(x)
        mean, h = torch.split(h, h.size(-1)//2, dim=-1)
        var = func.softplus(h) + 1e-8
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
            nn.Linear(z_dim, flattened_output/4),
            nn.LeakyReLU(),
            nn.Linear(flattened_output/4, flattened_output/2),
            nn.LeakyReLU(),
            nn.Linear(flattened_output/2, flattened_output/2),
            nn.LeakyReLU(),
            nn.Linear(flattened_output/2, flattened_output)
        )

    def forward(self, z):
        output = self._net(z)




class SimpleVae(BaseModel):
    def __init__(self):
        super().__init__()

