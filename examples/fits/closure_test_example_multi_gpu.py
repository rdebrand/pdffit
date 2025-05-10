import eko
import pineappl 
import numpy as np
import yaml

import torch
import lightning as L

torch.set_default_dtype(torch.float32)
pdffit_dvc = "cpu"
import pdffit

from matplotlib import pyplot as plt
from IPython.display import clear_output
from pdffit.examples.evolution.evo_bases import *

plt.rcParams.update({
	#"figure.dpi": 100,               
	#"savefig.dpi": 100,             
	"figure.figsize": (10, 6),
	'text.usetex': True,
	"font.family": 'Computer Modern',  
	"font.size": 15     
})

from validphys.api import API
from validphys.fkparser import load_fktable
from validphys import convolution
from validphys.core import PDF

# Define the input parameters
inp = {
	"dataset_input": {"dataset": "HERA_NC_318GEV_EM-SIGMARED", "variant": "legacy"},  # See https://inspirehep.net/literature/276661
	"use_cuts": "internal",
	"theoryid": 40000000,
}

# inp = {
# 	"dataset_input": {"dataset": "BCDMS_NC_NOTFIXED_P_EM-F2", "variant": "legacy_dw"},  # See https://inspirehep.net/literature/276661
# 	"use_cuts": "internal",
# 	"theoryid": 40000000,
# }

# Load the data using the validphys API
dataset = API.dataset(**inp)

# Covariance matrix for the SLAC data
covariance_matrix = API.covmat_from_systematics(**inp)

# NOTE: There is only one FK table for this dataset
FKTableData = load_fktable(*dataset.fkspecs).with_cuts(dataset.cuts)

xgrid = np.array(FKTableData.xgrid, dtype=np.float64)
xgrid_torch = torch.tensor(xgrid, dtype=torch.get_default_dtype())[:-1].detach()

# NOTE: flavors are in the evolution basis
ACTIVE_FLAVORS = FKTableData.luminosity_mapping

fk_array = FKTableData.get_np_fktable()
fk_array_torch = torch.tensor(fk_array, ).requires_grad_(False)
fk_array_torch = fk_array_torch[:, :, :-1]  # remove the last x, Q2, and flavor

def sort_rm(channels):
	"""
	TODO: docstring
	"""
	ev_pid = np.array(eko.basis_rotation.evol_basis_pids)
	idx = np.zeros(14, dtype=np.int32)
	for i, item in enumerate(channels):
		idx[i] = np.sum(np.where(ev_pid == item[0], np.arange(0, 14), 0))
	return idx


def sort_rm_new():
	"""
	TODO: docstring
	"""
	return eko.basis_rotation.rotate_flavor_to_evolution


def beta_pdf(x, b):
	"""
	TODO: docstring
	"""
	return torch.exp(b.log_prob(x))

def to_sigmared(fk, rotpdf):
	"""
	Forward map. Maps PDF to data space.
	
	Parameters
	----------
	fk: 
	
	rotpdf:
	
	Returns
	-------
	"""
	return (fk * rotpdf).sum((1,2))

def beta_w_vsr(alpha, n, delta):
	return (alpha-1)*(n/delta - 1)

def chi2_loss(d, t, Cinv):
	"""
	TODO: ...
	"""
	return (d-t).view(1,-1) @ (Cinv) @ (d-t)

rm_eko_flavor_evo = torch.tensor(eko.basis_rotation.rotate_flavor_to_evolution, ).to(torch.get_default_dtype())

evo_names = [r"$\gamma$", r"$\Sigma$", r"$g$", r"$V$", r"$V_3$", r"$V_8$", r"$V_{15}$", r"$V_{24}$", r"$V_{35}$", r"$T_3$", r"$T_8$", r"$T_{15}$", r"$T_{24}$", r"$T_{35}$"]
evo_reduced_names = [r"$\gamma$", r"$\Sigma$", r"$g$", r"$V = V_8 = V_{15}$", r"$V_3$", r"$V_8$", r"$V_{15}$", r"$V_{24}$", r"$V_{35}$", r"$T_3$", r"$T_8$", r"$T_{15}$", r"$T_{24}$", r"$T_{35}$"]

deltas_flavor = torch.zeros((14,), )

delta_uv = 0.27
delta_dv = 0.126
delta_db = 0.07  /2; deltas_flavor[6] = delta_db
delta_ub = 0.13  /2; deltas_flavor[5] = delta_ub
delta_g  = 0.4     ; deltas_flavor[7] = delta_g
delta_s  = 0.002 /2; deltas_flavor[4] = delta_s; deltas_flavor[10] = delta_s
delta_c  = 0.001 /2; deltas_flavor[3] = delta_c; deltas_flavor[11] = delta_c

delta_u = delta_uv + delta_ub; deltas_flavor[9] = delta_u
delta_d = delta_dv + delta_db; deltas_flavor[8] = delta_d

deltas_evo = torch.matmul(rm_eko_flavor_evo, deltas_flavor).detach()

beta_param_uv = beta_w_vsr(1.8, 2, delta_uv)
beta_param_dv = beta_w_vsr(2.2, 1, delta_dv)

beta_uv_in = torch.distributions.Beta(torch.tensor(1.8 ), torch.tensor(beta_param_uv))
beta_dv_in = torch.distributions.Beta(torch.tensor(2.2 ), torch.tensor(beta_param_dv))

beta_ub_in = torch.distributions.Beta(torch.tensor(0.81), torch.tensor(5.3))
beta_db_in = torch.distributions.Beta(torch.tensor(0.83), torch.tensor(4.8))
beta_g_in  = torch.distributions.Beta(torch.tensor(0.85), torch.tensor(4.7))
beta_s_in  = torch.distributions.Beta(torch.tensor(0.78), torch.tensor(5.2))
beta_c_in  = torch.distributions.Beta(torch.tensor(0.77), torch.tensor(4.9))

pdf_input = torch.zeros((14, xgrid_torch.shape[0]), )

pdf_input[9] = delta_uv * beta_uv_in.log_prob(xgrid_torch).exp() + delta_ub * beta_ub_in.log_prob(xgrid_torch).exp() #u
pdf_input[8] = delta_dv * beta_dv_in.log_prob(xgrid_torch).exp() + delta_db * beta_db_in.log_prob(xgrid_torch).exp() #d

pdf_input[5] = delta_ub * beta_ub_in.log_prob(xgrid_torch).exp() #ub
pdf_input[7] = delta_g  * beta_g_in.log_prob(xgrid_torch).exp() #g
pdf_input[6] = delta_db * beta_db_in.log_prob(xgrid_torch).exp() #db

pdf_input[10] = delta_s * beta_s_in.log_prob(xgrid_torch).exp() #s
pdf_input[4]  = delta_s * beta_s_in.log_prob(xgrid_torch).exp() #sb
pdf_input[11] = delta_c * beta_c_in.log_prob(xgrid_torch).exp() #c
pdf_input[3]  = delta_c * beta_c_in.log_prob(xgrid_torch).exp() #cb

# rotate the input pdf to the evolution basis

pdf_input_ev = torch.matmul(rm_eko_flavor_evo, pdf_input)

# Number of valence quarks for the evolution basis valence likes

val_num_03 = torch.tensor([3], requires_grad=False) #V
val_num_04 = torch.tensor([1], requires_grad=False) #V_3
val_num_05 = torch.tensor([3], requires_grad=False) #V_8
val_num_06 = torch.tensor([3], requires_grad=False) #V_15

val_nums = torch.tensor([3, 1, 3, 3], requires_grad=False)

L0_data = to_sigmared(fk_array_torch, pdf_input_ev[ACTIVE_FLAVORS]).requires_grad_(False)

Cinv = torch.tensor(np.linalg.inv(covariance_matrix), ).requires_grad_(False)

# %%
hidden_dim=128
in_out_dim=1

net_01     = pdffit.nn.DFF_f(in_out_dim, hidden_dim, trafo="sig", act="ct", n_hidden_l=2, w_init_=True) # singlet
net_02     = pdffit.nn.DFF_f(in_out_dim, hidden_dim, trafo="sig", act="ct", n_hidden_l=2, w_init_=True) # gluon
net_03     = pdffit.nn.DFF_f(in_out_dim, hidden_dim, trafo="sig", act="ct", n_hidden_l=2, w_init_=True) # V
net_04     = pdffit.nn.DFF_f(in_out_dim, hidden_dim, trafo="sig", act="ct", n_hidden_l=2, w_init_=True) # V_3
net_05     = pdffit.nn.DFF_f(in_out_dim, hidden_dim, trafo="sig", act="ct", n_hidden_l=2, w_init_=True) # V_8 
net_06     = pdffit.nn.DFF_f(in_out_dim, hidden_dim, trafo="sig", act="ct", n_hidden_l=2, w_init_=True) # V_15 
net_09     = pdffit.nn.DFF_f(in_out_dim, hidden_dim, trafo="sig", act="ct", n_hidden_l=2, w_init_=True) # T_3 
net_10     = pdffit.nn.DFF_f(in_out_dim, hidden_dim, trafo="sig", act="ct", n_hidden_l=2, w_init_=True) # T_8 
net_11     = pdffit.nn.DFF_f(in_out_dim, hidden_dim, trafo="sig", act="ct", n_hidden_l=2, w_init_=True) # T_15 

net_01_inv = pdffit.nn.DFF_g(in_out_dim, hidden_dim, trafo="sig") # singlet
net_02_inv = pdffit.nn.DFF_g(in_out_dim, hidden_dim, trafo="sig") # gluon
net_03_inv = pdffit.nn.DFF_g(in_out_dim, hidden_dim, trafo="sig") # V
net_04_inv = pdffit.nn.DFF_g(in_out_dim, hidden_dim, trafo="sig") # V_3
net_05_inv = pdffit.nn.DFF_g(in_out_dim, hidden_dim, trafo="sig") # V_8 
net_06_inv = pdffit.nn.DFF_g(in_out_dim, hidden_dim, trafo="sig") # V_15 
net_09_inv = pdffit.nn.DFF_g(in_out_dim, hidden_dim, trafo="sig") # T_3 
net_10_inv = pdffit.nn.DFF_g(in_out_dim, hidden_dim, trafo="sig") # T_8 
net_11_inv = pdffit.nn.DFF_g(in_out_dim, hidden_dim, trafo="sig") # T_15 

net_list = [net_01, net_02, net_03, net_04, net_05, net_06, net_09, net_10, net_11,
			net_01_inv, net_02_inv, net_03_inv, net_04_inv, net_05_inv, net_06_inv, net_09_inv, net_10_inv, net_11_inv]

# checkpoint_path = "pretrain_2_128.pt"
# checkpoint_path = "test_ltng.pt"

# checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# net_01.load_state_dict(checkpoint['net_01_state_dict'])
# net_02.load_state_dict(checkpoint['net_02_state_dict'])
# net_03.load_state_dict(checkpoint['net_03_state_dict'])
# net_04.load_state_dict(checkpoint['net_04_state_dict'])
# net_05.load_state_dict(checkpoint['net_05_state_dict'])
# net_06.load_state_dict(checkpoint['net_06_state_dict'])
# net_09.load_state_dict(checkpoint['net_09_state_dict'])
# net_10.load_state_dict(checkpoint['net_10_state_dict'])
# net_11.load_state_dict(checkpoint['net_11_state_dict']) 

# %%
s_0 = torch.tensor(0, requires_grad=False)
s_05 = torch.tensor(1/2, requires_grad=False)
s_08 = torch.tensor(0.8, requires_grad=False)
s_1 = torch.tensor(1, requires_grad=False)

fine_x_grid = torch.logspace(-7, -0.001, 10000, ).view(-1,1).detach()

# %%
class combine_models(torch.nn.Module):
	def __init__(self, nets):
		super().__init__()
		self.nets = torch.nn.ModuleList(nets)

	def construct_flows(self, xgrid_torch = xgrid_torch, fine_x_grid = fine_x_grid, val_alpha = torch.tensor(1.5), sea_alpha = torch.tensor(0.8), beta = torch.tensor(5.), 
			 val_beta_03 = beta_w_vsr(1.5, val_num_03.item(), deltas_evo[ 3]).detach(), 
			 val_beta_04 = beta_w_vsr(1.5, val_num_04.item(), deltas_evo[ 4]).detach(), 
			 val_beta_05 = beta_w_vsr(1.5, val_num_05.item(), deltas_evo[ 5]).detach(), 
			 val_beta_06 = beta_w_vsr(1.5, val_num_06.item(), deltas_evo[ 6]).detach()):

		self.base_val_01 = torch.distributions.Beta(val_alpha, beta)							  #singlet
		self.base_val_02 = torch.distributions.Beta(val_alpha, beta)							  #gluon
		self.base_val_03 = torch.distributions.Beta(val_alpha, val_beta_03) #V
		self.base_val_04 = torch.distributions.Beta(val_alpha, val_beta_04) #V_3
		self.base_val_05 = torch.distributions.Beta(val_alpha, val_beta_05) #V_8
		self.base_val_06 = torch.distributions.Beta(val_alpha, val_beta_06) #V_15
		self.base_val_09 = torch.distributions.Beta(val_alpha, beta)							  #T_3
		self.base_val_10 = torch.distributions.Beta(val_alpha, beta)							  #T_8
		self.base_val_11 = torch.distributions.Beta(val_alpha, beta)							  #T_15

		self.base_sea_01 = torch.distributions.Beta(sea_alpha, beta) #singlet
		self.base_sea_02 = torch.distributions.Beta(sea_alpha, beta) #gluon
		self.base_sea_03 = torch.distributions.Beta(sea_alpha, beta) #V
		self.base_sea_04 = torch.distributions.Beta(sea_alpha, beta) #V_3
		self.base_sea_05 = torch.distributions.Beta(sea_alpha, beta) #V_8
		self.base_sea_06 = torch.distributions.Beta(sea_alpha, beta) #V_15
		self.base_sea_09 = torch.distributions.Beta(sea_alpha, beta) #T_3
		self.base_sea_10 = torch.distributions.Beta(sea_alpha, beta) #T_8
		self.base_sea_11 = torch.distributions.Beta(sea_alpha, beta) #T_15

		self.flown_01_f = pdffit.nn.flown_comb_init(xgrid_torch.view(-1,1), self.base_val_01, self.base_sea_01) # singlet
		self.flown_02_f = pdffit.nn.flown_comb_init(xgrid_torch.view(-1,1), self.base_val_02, self.base_sea_02) # gluon
		self.flown_03_f = pdffit.nn.flown_comb_init(xgrid_torch.view(-1,1), self.base_val_03, self.base_sea_03) # V
		self.flown_04_f = pdffit.nn.flown_comb_init(xgrid_torch.view(-1,1), self.base_val_04, self.base_sea_04) # V_3
		self.flown_05_f = pdffit.nn.flown_comb_init(xgrid_torch.view(-1,1), self.base_val_05, self.base_sea_05) # V_8
		self.flown_06_f = pdffit.nn.flown_comb_init(xgrid_torch.view(-1,1), self.base_val_06, self.base_sea_06) # V_15
		self.flown_09_f = pdffit.nn.flown_comb_init(xgrid_torch.view(-1,1), self.base_val_09, self.base_sea_09) # T_3
		self.flown_10_f = pdffit.nn.flown_comb_init(xgrid_torch.view(-1,1), self.base_val_10, self.base_sea_10) # T_8
		self.flown_11_f = pdffit.nn.flown_comb_init(xgrid_torch.view(-1,1), self.base_val_11, self.base_sea_11) # T_15

		self.flown_03_grid_f = pdffit.nn.flown_comb_init(fine_x_grid.view(-1,1), self.base_val_03, self.base_sea_03) # V
		self.flown_04_grid_f = pdffit.nn.flown_comb_init(fine_x_grid.view(-1,1), self.base_val_04, self.base_sea_04) # V_3
		self.flown_05_grid_f = pdffit.nn.flown_comb_init(fine_x_grid.view(-1,1), self.base_val_05, self.base_sea_05) # V_8
		self.flown_06_grid_f = pdffit.nn.flown_comb_init(fine_x_grid.view(-1,1), self.base_val_06, self.base_sea_06) # V_15
	   
	def forward(self):

		flown_01 = self.flown_01_f(self.nets[ 0], 			 s_08) # singlet
		flown_02 = self.flown_02_f(self.nets[ 1], 			  s_0) # gluon
		flown_03 = self.flown_03_f(self.nets[ 2], 			  s_1) # V
		flown_04 = self.flown_04_f(self.nets[ 3], 			  s_1) # V_3
		# flown_05 = self.flown_05_f(self.nets[ 4], 			  s_1) # V_8
		# flown_06 = self.flown_06_f(self.nets[ 5], 			  s_1) # V_15
		flown_09 = self.flown_09_f(self.nets[ 6], 			  s_0) # T_3
		flown_10 = self.flown_10_f(self.nets[ 7], 			 s_08) # T_8
		flown_11 = self.flown_11_f(self.nets[ 8], 			 s_08) # T_15

		# flown_03_grid = self.flown_03_grid_f(self.nets[ 2], s_1) # V
		# flown_04_grid = self.flown_04_grid_f(self.nets[ 3], s_1) # V_3
		# flown_05_grid = self.flown_05_grid_f(self.nets[ 4], s_1) # V_8 = V
		# flown_06_grid = self.flown_06_grid_f(self.nets[ 5], s_1) # V_15 = V

		return torch.stack((flown_01, flown_02, flown_03, flown_04, flown_03, flown_03, flown_09, flown_10, flown_11), dim=0)
				#torch.stack((flown_03_grid, flown_04_grid, flown_03_grid, flown_03_grid), dim=0))
	
net_list[4] = torch.nn.Identity(); net_list[5] = torch.nn.Identity()

class LFlow(L.LightningModule):
	def __init__(self):
		super().__init__()

		self.flown_combined = combine_models(net_list[:9])

		self.register_buffer("val_alpha", torch.tensor(1.5))
		self.register_buffer("sea_alpha", torch.tensor(0.8))
		self.register_buffer("beta", torch.tensor(5.))
		self.register_buffer("val_beta_03", beta_w_vsr(1.5, val_num_03.item(), deltas_evo[ 3]).detach())
		self.register_buffer("val_beta_04", beta_w_vsr(1.5, val_num_04.item(), deltas_evo[ 4]).detach())
		self.register_buffer("val_beta_05", beta_w_vsr(1.5, val_num_05.item(), deltas_evo[ 5]).detach())
		self.register_buffer("val_beta_06", beta_w_vsr(1.5, val_num_06.item(), deltas_evo[ 6]).detach())

		self.register_buffer("fine_x_grid", fine_x_grid)

		self.register_buffer("devo", deltas_evo)
		self.register_buffer("ACTIVE_FLAVORS", torch.tensor(ACTIVE_FLAVORS, dtype = torch.int64))

		self.register_buffer("s_1", s_1)

		self.register_buffer("L0_data", L0_data)
		self.register_buffer("Cinv", Cinv)
		self.register_buffer("fk_array_torch", fk_array_torch)
		self.register_buffer("xgrid_torch", xgrid_torch)

		self.register_buffer("pdf_input_ev", pdf_input_ev)

	def on_fit_start(self):
		self.flown_combined.construct_flows(self.xgrid_torch, self.fine_x_grid, 
												self.val_alpha, self.sea_alpha, self.beta, 
												self.val_beta_03, self.val_beta_04, self.val_beta_05, self.val_beta_06)

	def training_step(self, batch, batch_idx):
		pdf_m_non_normalized = self.flown_combined() 

		pdf_m = self.devo[self.ACTIVE_FLAVORS].view(-1,1) * pdf_m_non_normalized
		# pdf_m[0] = self.pdf_input_ev[1]
		# pdf_m[6] = self.pdf_input_ev[9]
		pdf_m[7] = self.pdf_input_ev[10]
		pdf_m[8] = self.pdf_input_ev[11]
		# pdf_m[2] = self.pdf_input_ev[3]
		# pdf_m[1] = self.pdf_input_ev[2]
		loss_val_v = torch.dist(self.devo[3]*torch.trapezoid(pdf_m_non_normalized[3]/self.xgrid_torch, self.xgrid_torch), 3*self.s_1, 4)
		loss_val_v3 = torch.dist(self.devo[4]*torch.trapezoid(pdf_m_non_normalized[4]/self.xgrid_torch, self.xgrid_torch), self.s_1, 4)
		loss_chi2 = chi2_loss(to_sigmared(self.fk_array_torch, pdf_m), self.L0_data, self.Cinv)
		loss = loss_chi2 + loss_chi2.detach()*(3*loss_val_v + loss_val_v3)
		return loss

	def configure_optimizers(self):

		act_pars_f, other_pars_f = ([], [])

		for f in [self.flown_combined.nets[i] for i in [0,1,2,3,6,7,8]]:
			act_pars_f_ = []
			other_pars_f_ = []
			for module in f.arc:
				if isinstance(module, pdffit.CT):
					act_pars_f_ += list(module.parameters())
				else:
					other_pars_f_ += list(module.parameters())
			act_pars_f += act_pars_f_
			other_pars_f.append(other_pars_f_)

		opt = torch.optim.Adam([
		{'params': other_pars_f[ 0], 'lr':1e-4}, #singlet
		{'params': other_pars_f[ 1], 'lr':1e-4}, #gluon
		{'params': other_pars_f[ 2], 'lr':1e-4}, #V
		{'params': other_pars_f[ 3], 'lr':1e-4}, #V_3
		# {'params': other_pars_f[ 4], 'lr':5e-5}, #V_8
		# {'params': other_pars_f[ 5], 'lr':5e-5}, #V_15
		{'params': other_pars_f[ 4], 'lr':1e-4}, #T_3
		{'params': other_pars_f[ 5], 'lr':1e-4}, #T_8
		{'params': other_pars_f[ 6], 'lr':1e-4}, #T_15
		{'params': act_pars_f, 'lr':1e-4}])
		return opt

ltng_flow = torch.compile(LFlow())

trainer = L.Trainer(max_epochs = 1, accelerator="gpu", devices = 8)
train_loader = torch.utils.data.DataLoader(torch.ones((1000000,)), num_workers=0)

torch.set_float32_matmul_precision('medium')
if __name__ == "__main__":
    trainer.fit(model=ltng_flow, train_dataloaders= train_loader)

torch.save({
		'net_01_state_dict': net_01.state_dict(),
		'net_02_state_dict': net_02.state_dict(),
		'net_03_state_dict': net_03.state_dict(),
		'net_04_state_dict': net_04.state_dict(),
		'net_05_state_dict': net_05.state_dict(),
		'net_06_state_dict': net_06.state_dict(),
		'net_09_state_dict': net_09.state_dict(),
		'net_10_state_dict': net_10.state_dict(),
		'net_11_state_dict': net_11.state_dict()},
		"test_ltng.pt")