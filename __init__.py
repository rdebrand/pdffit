import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from . import nn
from .nn_classes import CT
from .viz.plot_utils import plot_densities, plot_trafo
from .train_utils import train

__all__ = ["device"]