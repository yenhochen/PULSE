# import copy
# from glob import glob

import abc
import torch
import torch.nn.functional as F

from torch import nn


class Reconstruction(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def reshape_output_params(self, output_params):
        pass

    @abc.abstractmethod
    def compute_loss(self, data, output_params):
        pass

    @abc.abstractmethod
    def compute_means(self, output_params):
        pass


class Gaussian(nn.Module, Reconstruction):
    def __init__(self):
        super().__init__()
        self.n_params = 2

    def reshape_output_params(self, output_params):
        means, logvars = torch.chunk(output_params, 2, -1)
        return torch.stack([means, logvars], -1)

    def compute_loss(self, data, output_params):
        means, logvars = torch.unbind(output_params, axis=-1)
        recon_all = F.gaussian_nll_loss(
            input=means, target=data, var=torch.exp(logvars), reduction="none"
        )
        return recon_all

    def compute_means(self, output_params):
        return output_params[..., 0]
