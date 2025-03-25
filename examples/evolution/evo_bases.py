import torch
from eko.basis_rotation import rotate_flavor_to_evolution as rfe
from pdffit import device

# define the base beta parameters for a stacked distribtion of normalized to 1
#			  'ph', 'tbar', 'bbar', 'cbar', 'sbar', 'ubar', 'dbar', 'g',  'dv', 'uv', 's', 'c', 'b', 't'
base_alphas = [0.8,  0.8,    0.8,    0.8,    0.8,    0.8  ,  0.8,    0.8 , 2.0,  2.0,  0.8, 0.8, 0.8, 0.8]
base_betas  = [5.0,  5.0,    5.0,    5.0,    5.0,    5.0  ,  5.0,    5.0 , 7.0,  7.0,  5.0, 5.0, 5.0, 5.0]

in_alphas =   [0.8,  0.8,    0.8,    0.8,    0.8,    0.7  ,  0.6,    0.85, 2.2,  1.8,  0.8, 0.8, 0.8, 0.8]
in_betas  =   [5.0,  5.0,    5.0,    5.0,    5.0,    5.3  ,  4.8,    4.7 , 6.7,  7.4,  5.0, 5.0, 5.0, 5.0]

stacked_base_dist = torch.distributions.Beta(torch.tensor(base_alphas, device=device, dtype=torch.float64), torch.tensor(base_betas, device=device, dtype=torch.float64))
stacked_in_dist   = torch.distributions.Beta(torch.tensor(in_alphas  , device=device, dtype=torch.float64), torch.tensor(in_betas  , device=device, dtype=torch.float64))

# define the different momenta for the flavors 
# in this parametrization sum(deltas) = 1 - delta_dbar - delta_ubar, because of uv and dv parametrization
#			  'ph', 'tbar', 'bbar', 'cbar', 'sbar', 'ubar', 'dbar', 'g',  'dv', 'uv', 's', 'c', 'b', 't'
deltas_vals = [0.0,  0.0,    0.0,    0.0,    0.0,    0.075,  0.05,   0.45, 0.1,  0.2,  0.0, 0.0, 0.0, 0.0]
deltas_m = torch.tensor(deltas_vals, device=device, dtype=torch.float64) * torch.eye(14, device=device, dtype=torch.float64)

# rotation matrix to pure flavor
#'ph', 'tbar', 'bbar', 'cbar', 'sbar', 'ubar', 'dbar', 'g', 'd', 'u', 's', 'c', 'b', 't'
to_pure_flavor = torch.eye(14, device=device, dtype=torch.float64); to_pure_flavor[8, 6] = 1.0; to_pure_flavor[9, 5] = 1.0

# rotation from pure flavor to evolution basis
# from: 'ph', 'tbar', 'bbar', 'cbar', 'sbar', 'ubar', 'dbar', 'g'  , 'd'  , 'u' , 's' , 'c'  , 'b'  , 't'
# to  : 'ph', 'S'   , 'g'   , 'V'   , 'V3'  , 'V8'  , 'V15' , 'V24', 'V35', 'T3', 'T8', 'T15', 'T24', 'T35'
rm_pflavor_evo = torch.tensor(rfe, device=device, dtype=torch.float64)

# then rotation from our parametrization to evo delta basis
# from: 'ph', 'tbar', 'bbar', 'cbar', 'sbar', 'ubar', 'dbar', 'g'  , 'dv'  , 'uv' , 's' , 'c'  , 'b'  , 't'
# to  : 'ph', 'S'   , 'g'   , 'V'   , 'V3'  , 'V8'  , 'V15' , 'V24', 'V35', 'T3', 'T8', 'T15', 'T24', 'T35'
rm_flavor_evo = (rm_pflavor_evo@to_pure_flavor@deltas_m).detach()

class evo_bases:
   
    def __init__(self, alphas, betas, deltas):
        self.stacked_betas = torch.distributions.Beta(torch.tensor(alphas,  device=device, dtype=torch.float64), 
                                                torch.tensor(betas, device=device, dtype=torch.float64))
        self.deltas = deltas
        self.jac_fac = self.delta_rfe.sum(-1)
        self.rfe = rm_flavor_evo

    def evo_log_prob(self, x, idx):
        return (torch.dot(self.rfe[idx], self.stacked_betas.log_prob(x).exp())).log(), self.jac_fac[idx]