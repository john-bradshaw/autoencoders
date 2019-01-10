
import torch
from torch import nn

from .dist_parameterisers.base_parameterised_distribution import BaseParameterisedDistribution


class SingleLatentWithPriorAE(nn.Module):
    def __init__(self,
                 encoder: BaseParameterisedDistribution,
                 decoder: BaseParameterisedDistribution,
                 latent_prior: BaseParameterisedDistribution):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_prior = latent_prior
        self._tb_logger = None

    def forward(self, x, beta):
        raise NotImplementedError

    def reconstruct_no_grad(self, x, sample_z=False, sample_x=False):
        with torch.no_grad():
            z = self._run_through_to_z(x, sample_z)
            x = self.decode_from_z_no_grad(z, sample_x)
        return x

    def nll_from_z_no_grad(self, z, x):
        with torch.no_grad():
            self.decoder.update(z)
            nll = self.decoder.nlog_like_of_obs(x)
        return nll

    def sample_from_prior_no_grad(self, sample_x=False):
        with torch.no_grad():
            z = self.latent_prior.sample_no_grad(1)[0]
            x = self.decode_from_z_no_grad(z, sample_x)
        return x

    def decode_from_z_no_grad(self, z, sample_x=False):
        with torch.no_grad():
            self.decoder.update(z)
            x = self.decoder.sample_no_grad(1)[0] if sample_x else self.decoder.mode()
        return x

    def _run_through_to_z(self, x, sample_z=False):
        self.encoder.update(x)
        z = self.encoder.sample_no_grad(1)[0] if sample_z else self.encoder.mode()
        return z