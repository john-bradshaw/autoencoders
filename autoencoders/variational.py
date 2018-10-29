
import torch
from torch import nn

from .dist_parameterisers.base_parameterised_distribution import BaseParameterisedDistribution


class VAE(nn.Module):
    """
    Basic Variational Autoencoder

    ::
        @ARTICLE{Kingma2013-it,
          title         = "{Auto-Encoding} Variational Bayes",
          author        = "Kingma, Diederik P and Welling, Max",
          abstract      = "How can we perform efficient inference and learning in
                           directed probabilistic models, in the presence of continuous
                           latent variables with intractable posterior distributions,
                           and large datasets? We introduce a stochastic variational
                           inference and learning algorithm that scales to large
                           datasets and, under some mild differentiability conditions,
                           even works in the intractable case. Our contributions is
                           two-fold. First, we show that a reparameterization of the
                           variational lower bound yields a lower bound estimator that
                           can be straightforwardly optimized using standard stochastic
                           gradient methods. Second, we show that for i.i.d. datasets
                           with continuous latent variables per datapoint, posterior
                           inference can be made especially efficient by fitting an
                           approximate inference model (also called a recognition
                           model) to the intractable posterior using the proposed lower
                           bound estimator. Theoretical advantages are reflected in
                           experimental results.",
          month         =  dec,
          year          =  2013,
          archivePrefix = "arXiv",
          primaryClass  = "stat.ML",
          eprint        = "1312.6114v10"
        }
    """
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
        """
        convenience function calculates the ELBO term
        """
        return self.elbo(x, beta)

    def elbo(self, x, beta=1.):
        self.encoder.update(x)
        z_sample = self.encoder.sample_via_reparam(1).squeeze(1)

        self.decoder.update(z_sample)
        log_like = -self.decoder.nlog_like_of_obs(x)

        elbo = log_like

        if beta != 0.:
            kl_term = -self.encoder.kl_with_other(self.latent_prior)
            elbo += beta * kl_term

            if self._tb_logger is not None:
                self.tb_logger.add_scalar('kl_term(no_beta)', kl_term)

        if self._tb_logger is not None:
            self.tb_logger.add_scalar('reconstruction_term', log_like)
            self.tb_logger.add_scalar('elbo', elbo)
        return elbo

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
            z = self.latent_prior.sample_no_grad(1).squeeze(1)
            x = self.decode_from_z_no_grad(z, sample_x)
        return x

    def decode_from_z_no_grad(self, z, sample_x=False):
        with torch.no_grad():
            self.decoder.update(z)
            x = self.decoder.sample_no_grad(1).squeeze(1) if sample_x else self.decoder.mode()
        return x

    def _run_through_to_z(self, x, sample_z=False):
        self.encoder.update(x)
        z = self.encoder.sample_no_grad(1).squeeze(1) if sample_z else self.encoder.mode()
        return z




