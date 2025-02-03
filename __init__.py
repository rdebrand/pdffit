import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from . import nn
from .viz.plot_utils import plot_densities, plot_trafo
from .train_utils import train

__all__ = ["device"]