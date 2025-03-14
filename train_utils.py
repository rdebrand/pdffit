import numpy as np

from . import torch, device
from .sampling import *
from .nn_classes import CT

import torch.optim as optim


def loss_nll(p, out, jac):
    return -p.log_prob(out)-torch.log(torch.abs(jac))

def train(f, 
		  g, 
		  p_z0, 
		  p_z1, 
		  lr, 
		  iters, 
		  bs = 2048, 
		  gamma = (False,), 
		  sampling=("dist",), 
		  print_itr = 250,
		  fast_jac = True,
		  loss_out = False,
		  lr_g = 1e-3,
		  g_only = False,
		  lr_act = None,
		  ):
	beta = t_one

	loss_arr = np.empty((3, iters))

	if lr_act is None:
		opt_f = optim.Adam(f.parameters(), lr=lr)
		opt_g = optim.Adam(g.parameters(), lr=lr_g)

	else:
		act_pars_f, other_pars_f, act_pars_g, other_pars_g = ([], [], [], [])

		for module in f:
			if isinstance(module, CT):
				act_pars_f += list(module.parameters())
			else:
				other_pars_f += list(module.parameters())

		for module in g:
			if isinstance(module, CT):
				act_pars_g += list(module.parameters())
			else:
				other_pars_g += list(module.parameters())

		opt_f = optim.Adam([
			{'params': other_pars_f, 'lr': lr},
			{'params': act_pars_f, 'lr': lr_act}
		])
		opt_g = optim.Adam([
			{'params': other_pars_g, 'lr': lr_g},
			{'params': act_pars_g, 'lr': lr_act}
		])
	
	if gamma[0]:
		opt_gamma = optim.Adam((gamma[1],), lr=lr_g)
	else:
		loss_gamma = torch.tensor(0., dtype=torch.float32, device=device)

	for itr in range(1, iters+1):
		opt_f.zero_grad()
		opt_g.zero_grad()

		if sampling[0] == "dist":
			x = get_batch(p_z1, bs)		
		elif sampling[0] == "hist":
			x = sample_hist(sampling[1], bs, p_z1).requires_grad_()

		if fast_jac:
			z, jac = f(x)
		else:
			z = f(x)
			jac = torch.autograd.grad(z[:,0].sum(), x, create_graph=True)[0]

		x_ = g(z.detach())
		loss_nll_ = loss_nll(p_z0, z, jac)
		loss_reco = torch.dist(x_, x.detach())
		loss = (loss_nll_ + beta*loss_reco).mean()
		if gamma[0]:
			loss += torch.dist((x_.detach()**gamma[1]).mean(), gamma[2].detach())


		loss.backward()
		if not g_only:
			opt_f.step()
		opt_g.step()

		if gamma[0]:
			opt_gamma.step()
			opt_gamma.zero_grad()

		if itr % print_itr == 0:
			if itr == 1000:
				opt_f.param_groups[0]['lr'] = 1e-3
				opt_g.param_groups[0]['lr'] = 1e-3
			if itr == 2200:
				opt_f.param_groups[0]['lr'] = 1e-4
				opt_g.param_groups[0]['lr'] = 1e-4
			if itr == 3500:
				opt_f.param_groups[0]['lr'] = 1e-5
				opt_g.param_groups[0]['lr'] = 1e-5
			print(itr, loss.item(), loss_reco.mean().item(), loss_nll_.mean().item())

		loss_arr[0, itr-1] = loss_nll_.mean().item()
		loss_arr[1, itr-1] = beta*loss_reco.mean().item()
	if loss_out:
		return loss_arr