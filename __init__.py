import torch

try: 
    from __main__ import pdffit_dvc as device
except:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from . import nn
from .nn_classes import CT
from .viz.plot_utils import plot_densities, plot_trafo
from .train_utils import train