from .analytical import *
from .nn_classes import lgnn, clamp, Log10, Exp10
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


class DFF_f(nn.Module):
	def __init__(self, in_out_dim, hidden_dim, trafo = "sig"):
		super().__init__()
		
		self.in_out_dim = in_out_dim
		self.hidden_dim = hidden_dim
		
		self.arc = nn.Sequential(
								nn.Linear(in_out_dim,hidden_dim),
								nn.Linear(hidden_dim,hidden_dim),
								nn.Linear(hidden_dim,in_out_dim)
								)
		
		if trafo == "sig":
			self.f_fwd = f_sig
			self.arc.apply(weights_init)

		if trafo == "log":
			self.f_fwd = f_log
		
	def forward(self, x):
		return self.f_fwd(x, self.arc)
	

class DFF_g(nn.Module):
	def __init__(self, in_out_dim, hidden_dim, trafo = "sig"):
		super().__init__()
		
		self.in_out_dim = in_out_dim
		self.hidden_dim = hidden_dim

		if trafo == "sig":
			expand, compress = (lgnn(), nn.Sigmoid())

		if trafo == "log":
			expand, compress = (Log10(), (Exp10(), clamp()))
		
		self.arc = nn.Sequential(expand, 
								nn.Linear(in_out_dim, hidden_dim),
								nn.Mish(),
								nn.Linear(hidden_dim, hidden_dim),
								nn.Mish(),
								nn.Linear(hidden_dim,in_out_dim), 
								*compress)
		
	def forward(self, x):
		return self.arc(x)

	

def flown(x, nn_, base, gamma = 1):
    z1 = torch.tensor(x, requires_grad=True).reshape(-1,1).to(device)
    z0 = nn_(z1**(1/gamma))
    return base.log_prob(z0).view(-1) + torch.log(torch.abs(torch.autograd.grad(torch.sum(z0), z1)[0].view(-1)))