from .analytical import *
from .nn_classes import lgnn, clamp, Log10, Exp10, CT as mod_CT
import torch.nn as nn
from torch import addmm, matmul, mul


def weights_init(m):
	if isinstance(m, nn.Linear):
		std = 2/(m.in_features + m.out_features)
		torch.nn.init.normal_(m.weight, mean = 3e-2, std=std)


def f_sig(input, nn_):
	lg, dlg = logit(input)
	l1 = addmm(nn_[0].bias, lg, nn_[0].weight.T)
	m1, dm1 = mish(l1)
	l2 = addmm(nn_[1].bias, m1, nn_[1].weight.T)
	m2, dm2 = mish(l2)
	l3 = addmm(nn_[2].bias, m2, nn_[2].weight.T)
	sm, dsm = sigmoid(l3)
	inoutgrad = mul(dlg, dsm)
	jac = matmul(matmul(inoutgrad*dm2*nn_[2].weight, nn_[1].weight)*dm1, nn_[0].weight)
	return sm, jac

clp = clamp()

def f_log(input, nn_):
	lg10, dlg10 = log10(input)
	l1 = addmm(nn_[0].bias, lg10, nn_[0].weight.T)
	m1, dm1 = mish(l1)
	l2 = addmm(nn_[1].bias, m1, nn_[1].weight.T)
	m2, dm2 = mish(l2)
	l3 = addmm(nn_[2].bias, m2, nn_[2].weight.T)
	exp, dexp = exp10(l3)
	inoutgrad = mul(dlg10, dexp)
	jac = matmul(matmul(inoutgrad*dm2*nn_[2].weight, nn_[1].weight)*dm1, nn_[0].weight)
	return clp(exp), jac

def f_ct_sig(input, nn_):
	lg, dlg = logit(input)
	l1 = addmm(nn_[0].bias, lg, nn_[0].weight.T)
	ct1, dct1 = ana_CT(l1, nn_[1].beta)
	l2 = addmm(nn_[2].bias, ct1, nn_[2].weight.T)
	ct2, dct2 = ana_CT(l2, nn_[3].beta)
	l3 = addmm(nn_[4].bias, ct2, nn_[4].weight.T)
	sm, dsm = sigmoid(l3)
	inoutgrad = mul(dlg, dsm)
	jac = matmul(matmul(inoutgrad*dct2*nn_[4].weight, nn_[2].weight)*dct1, nn_[0].weight)
	return sm, jac


class DFF_f(nn.Module):
	def __init__(self, in_out_dim, hidden_dim, trafo = "sig", act = "mish", w_init_ = True):
		super().__init__()
		
		self.in_out_dim = in_out_dim
		self.hidden_dim = hidden_dim
		
		if act == "mish":
			self.arc = nn.Sequential(
									nn.Linear(in_out_dim,hidden_dim),
									nn.Linear(hidden_dim,hidden_dim),
									nn.Linear(hidden_dim,in_out_dim)
									)
			
			if trafo == "sig":
				self.f_fwd = f_sig
				if w_init_:
					self.arc.apply(weights_init)

			if trafo == "log":
				self.f_fwd = f_log
		
		elif act == "ct":
			self.arc = nn.Sequential(
								nn.Linear(in_out_dim,hidden_dim),
								mod_CT(grad = True),
								nn.Linear(hidden_dim,hidden_dim),
								mod_CT(grad = True),
								nn.Linear(hidden_dim,in_out_dim)
								)
			self.f_fwd = f_ct_sig
		
	def forward(self, x):
		return self.f_fwd(x, self.arc)
	

class DFF_g(nn.Module):
	def __init__(self, in_out_dim, hidden_dim, trafo = "sig", act = "mish"):
		super().__init__()
		
		self.in_out_dim = in_out_dim
		self.hidden_dim = hidden_dim

		if trafo == "sig":
			expand, compress = (lgnn(), (nn.Sigmoid(),))

		if trafo == "log":
			expand, compress = (Log10(), (Exp10(), clamp()))
		
		if act == "mish":

			self.arc = nn.Sequential(expand, 
									nn.Linear(in_out_dim, hidden_dim),
									nn.Mish(),
									nn.Linear(hidden_dim, hidden_dim),
									nn.Mish(),
									nn.Linear(hidden_dim,in_out_dim), 
									*compress)
		
		elif act == "ct":
		
			self.arc = nn.Sequential(expand, 
									nn.Linear(in_out_dim, hidden_dim),
									mod_CT(),
									nn.Linear(hidden_dim, hidden_dim),
									mod_CT(),
									nn.Linear(hidden_dim,in_out_dim), 
									*compress)
		
	def forward(self, x):
		return self.arc(x)

	

def flown(z1, nn_, base, gamma = 1, nd = False):
	if gamma == 1:
		z1_shifted = z1
	else:
		z1_shifted = z1**(1/gamma)
	z0, jac = nn_(z1_shifted)
	if not nd:
		return base.log_prob(z0).view(-1) + torch.log(torch.abs(jac.view(-1)))
	else:
		return base.log_prob(z0).view(-1) + torch.log(torch.abs(jac.view(-1))) - torch.log(z1_shifted.view(-1))