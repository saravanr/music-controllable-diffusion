from typing import Any

import torch
import wandb
from torch.nn import functional as func
import numpy as np
import utils.model_utils as ut
from data.midi_data_module import MAX_MIDI_ENCODING_ROWS
from models.simple_vae.simple_vae import SimpleVae
import torch.nn.functional as F
wandb.init(project="music-controllable-diffusion-midi-gmiw_vae", entity="saravanr")


class GMIWVae(SimpleVae):
    def __init__(self, z_dim=64, k=50, input_shape=(10000, 8), alpha=1.0, *args: Any, **kwargs: Any):
        super(GMIWVae, self).__init__(z_dim=z_dim, input_shape=input_shape, alpha=1.0, *args, **kwargs)

        self._model_prefix = "GMIWVae"
        self._k = k

        # Mixture of Gaussians Prior
        self.z_pre = torch.nn.Parameter(
            torch.randn(1, 2 * self._k, self._z_dim) /
            np.sqrt(self._k * self._z_dim)
        )

        # Uniform weighting
        self.pi = torch.nn.Parameter(torch.ones(k) / k, requires_grad=False)

    @staticmethod
    def duplicate(x, rep):
        y = x.expand(rep, *x.shape).reshape(-1, *x.shape[1:])
        return y

    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs should all be scalar
        ################################################################################
        # We provide the learnable prior for you. Familiarize yourself with
        # this object by checking its shape.
        prior = ut.gaussian_parameters(self.z_pre, dim=1)
        batch = x.shape[0]

        # Duplicate x  m times
        weighted_x = ut.duplicate(x, iw)

        data = weighted_x.to('cpu').numpy()
        # Obtain the mean and var from encoder
        mean, log_var = self._encoder(x)
        log_var = F.softplus(log_var) + 1e-8

        # Duplicate mean and var
        weighted_m = ut.duplicate(mean, iw)
        weighted_log_v = ut.duplicate(log_var, iw)

        # We now have 'm' mean and variances. Sample `z`
        #z = ut.sample_gaussian(weighted_m, torch.exp(weighted_log_v))
        z = SimpleVae.reparameterize(weighted_m, weighted_log_v)

        # Obtain logits
        x_hat = self._decoder(z)

        # Obtain weighted priors
        num_to_expand = batch * iw
        weighted_p_m = prior[0].expand(num_to_expand, *prior[0].shape[1:])
        weighted_p_v = prior[1].expand(num_to_expand, *prior[1].shape[1:])
        weighted_p_v = F.softplus(weighted_p_v) + 1e-8

        z_weighted_priors = ut.log_normal_mixture(z, weighted_p_m, weighted_p_v)

        # Evaluate
        #reconstruction_loss = func.mse_loss(x_hat, x.view(-1, MAX_MIDI_ENCODING_ROWS * 8), reduction='sum')
        reconstruction_loss = -func.binary_cross_entropy(x_hat, weighted_x.view(-1, 784), reduction='mean')
        x_posteriors = reconstruction_loss
        z_posteriors = ut.log_normal(z, weighted_m, weighted_log_v)

        log_ratios = z_weighted_priors + x_posteriors - z_posteriors
        log_ratios = log_ratios.reshape(iw, batch)

        niwaes = ut.log_mean_exp(log_ratios, 0)
        niwae = -1.0 * torch.mean(niwaes)

        kls = z_posteriors - z_weighted_priors
        kl = torch.mean(kls)
        return niwae, kl, reconstruction_loss

    @staticmethod
    def detach_torch_tuple(args):
        return (v.detach() for v in args)

    @staticmethod
    def compute_metrics(fn, repeat, xl):
        metrics = [0, 0, 0]
        for _ in range(repeat):
            niwae, kl, rec = GMIWVae.detach_torch_tuple(fn(xl))
            metrics[0] += niwae / repeat
            metrics[1] += kl / repeat
            metrics[2] += rec / repeat
        return metrics

    def forward(self, x):
        for iw in [1, 10, 100]:
            repeat = max(100 // iw, 1)  # Do at least 100 iterations
            fn = lambda x: model.negative_iwae_bound(x, iw)
            niwae, kl, rec = GMIWVae.compute_metrics(fn, repeat, x)
            print("Negative IWAE-{}: {}".format(iw, niwae))


        niwae, kl, rec = self.negative_iwae_bound(x, iw=10)
        return niwae

    def step(self, batch, batch_idx):

        loss = self(batch)
        return loss


if __name__ == "__main__":
    print(f"Training GMVAE")
    batch_size = 2048
    train_mnist = True
    if train_mnist:
        model = GMIWVae(
            alpha=1,
            k=10,
            z_dim=20,
            input_shape=(28, 28),
            use_mnist_dms=True,
            sample_output_step=10,
            batch_size=batch_size
        )
    else:
        model = GMIWVae(
            alpha=1,
            z_dim=800,
            input_shape=(MAX_MIDI_ENCODING_ROWS, 8),
            use_mnist_dms=False,
            sample_output_step=10,
            batch_size=batch_size
        )
    print(f"Training --> {model}")

    max_epochs = 10000
    wandb.config = {
        "learning_rate": model.lr,
        "epochs": max_epochs,
        "batch_size": batch_size,
        "alpha": model.alpha
    }

    _optimizer = model.configure_optimizers()
    for epoch in range(1, max_epochs + 1):
        model.fit(epoch, _optimizer)
        if epoch % 1 == 0:
            # model.test()
            model.sample_output(epoch)
            model.save(epoch)
