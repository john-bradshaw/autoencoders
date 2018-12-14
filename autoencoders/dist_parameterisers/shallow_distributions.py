
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from .. import kl_div
from .. import settings
from .base_parameterised_distribution import BaseParameterisedDistribution

sett = settings.get_settings_manager()


class ShallowDistributions(BaseParameterisedDistribution):
    """
    Shallow distributions do not include any layers. They are directly paramterized distributions by a Tensor.
    """
    def __init__(self, parameterisation: torch.Tensor=None):
        """
        :param parameterisation: [b, ...]
        """
        super().__init__()
        self._params = parameterisation

    def update(self, x):
        self._params = x


class IndependentGaussianDistribution(ShallowDistributions):
    """
    Standard independent Gaussian distribution used eg for latents in the original VAE.
    The associated prior is the standard normal one.

    mean parameterised by the first half of parameters in final dimension. The log of the variance by the second half.

    """

    @property
    def mean_log_var(self):
        params = self._params
        split_point = params.shape[-1] // 2
        return params[..., :split_point], params[..., split_point:]

    def sample_via_reparam(self, num_samples: int=1) -> torch.Tensor:
        mean, log_var = self.mean_log_var
        std_dev = torch.exp(0.5 * log_var)

        samples = mean.unsqueeze(1) + torch.randn(log_var.shape[0], num_samples, *log_var.shape[1:],
                                     dtype=std_dev.dtype, device=std_dev.device) * std_dev.unsqueeze(1)
        return samples

    def mode(self) -> torch.Tensor:
        mean, _ = self.mean_log_var
        return mean

    def kl_with_other(self, other):
        if isinstance(other, IndependentGaussianDistribution):
            if (other._params == 0).all():
                mean, log_var = self.mean_log_var
                return kl_div.gauss_kl_with_std_norm(mean, log_var, reduce_fully=False)
        super().kl_with_other(other)

    def nlog_like_of_obs(self, obs: torch.Tensor) -> torch.Tensor:
        params = self._params
        if (params == 0).all():
            sq_error = obs*2
            term1 = 0.5 * np.log(2 * np.pi).astype(sett.np_float_type)
            nll = sq_error/2 + term1
        else:
            mean, log_var = self.mean_log_var
            sq_error = (obs - mean)**2
            term1 = 0.5 * np.log(2 * np.pi).astype(sett.np_float_type)
            term2 = 0.5 * log_var
            term3 = sq_error / (2 * torch.exp(log_var))
            nll = term2 + term3 + term1
        nll = nll.sum(dim=tuple(range(1, len(params.shape))))
        return nll  # sum over all but batch.


class BernoulliOnLogits(ShallowDistributions):
    """
    Independent Bernoulli distribution.
    Paramterers are logits, ie go through sigmoid to parameterise Bernoulli
    """


    def bernoulli_params(self):
        params = self._params
        return F.sigmoid(params)

    def sample_via_reparam(self, num_samples: int = 1) -> torch.Tensor:
        raise RuntimeError("Reparameterisation trick not applicable for Bernoulli distribution")

    def sample_no_grad(self, num_samples: int = 1) -> torch.Tensor:
        params = self._params
        params = params.unsqueeze(1).repeat(1, num_samples, *[1 for _ in params.shape[1:]])
        return torch.bernoulli(params)

    def mode(self) -> torch.Tensor:
        params = self._params
        return (params > 0.5).type(params.dtype)

    def kl_with_other(self, other):
        raise NotImplementedError("KL for Bernoulli not yet implemented -- need to be careful when distributions do not overlap")

    def nlog_like_of_obs(self, obs: torch.Tensor) -> torch.Tensor:
        params = self._params
        return F.binary_cross_entropy_with_logits(params, obs, reduction='none'
                                                  ).sum(dim=tuple(range(1, len(params.shape))))

