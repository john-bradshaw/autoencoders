
import abc
import typing

import torch
from torch import nn

T = typing.TypeVar('T', bound='BaseParameterisedDistribution')


class BaseParameterisedDistribution(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self, *input, **kwargs) -> T:
        raise NotImplementedError

    def forward(self, *input, **kwargs):
        """
        If in training mode will sample via reparameterisation trick. if eval mode then will return distribution mode.
        """
        num_samples = kwargs.pop('num_samples', 1)
        self.update(*input, **kwargs)
        if self.training:
            return self.sample_via_reparam(num_samples=num_samples).squeeze(1)
        else:
            return self.mode()

    def sample_via_reparam(self, num_samples: int=1) -> list:
        """
        Samples this distribution using re-paramterisation trick
        :return: num_samples samples for in a list [ (b,*sample.shape), ...]
        """
        raise NotImplementedError

    def sample_no_grad(self, num_samples: int=1) -> list:
        """
        Samples this distribution with no gradients flowing back.
        """
        with torch.no_grad():
            return self.sample_via_reparam(num_samples)

    def mode(self) -> torch.Tensor:
        """
        returns the mdoe of the parameterised distribution
        :return: [b, ...]
        """
        raise NotImplementedError

    def kl_with_other(self, other) -> torch.Tensor:
        """
        compute the KL divergence for each member of batch with other.
        :return: [b]
        """
        raise NotImplementedError

    def nlog_like_of_obs(self,obs: torch.Tensor) -> torch.Tensor:
        """
        compute the negative log likelihood of the sample under this dist.
        :return: [b]
        """
        raise NotImplementedError

