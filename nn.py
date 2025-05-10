from .analytical import *
from .nn_classes import lgnn, clamp, Log10, Exp10, CT as mod_CT
import torch.nn as nn
from torch import addmm, matmul, mul
from . import device

class DFF_f(nn.Module):
	"""
	Torch module to initialize the parameters for the 
	backward neural network.

	Args:
		in_out_dim (Int): Dimension of input and ouput. Default: ``1``.
		hidden_dim (Int): Dimension of input and ouput. Default: ``64``.
		trafo (str): Which expansion/compression transformation to use. Default: ``"sig"``.
		act (str): Activation functions between the layers. Default: ``"mish"``.
		w_init (Bool): If ``True`` use custom weight initialization,
			else use the torch default.
		w_init_mean (Float): The mean for the weight initialization Gaussian.
		n_hidden_l (Int): Number of hidden layers. Default: ``1``.
	"""

	def __init__(self, in_out_dim = 1, hidden_dim = 64, trafo = "sig", act = "mish", w_init_ = True, w_init_mean = 4e-2, n_hidden_l = 1, res = False):
		super().__init__()
		
		self.in_out_dim = in_out_dim
		self.hidden_dim = hidden_dim
		self.weights_init = weights_init_const(w_init_mean)
		
		if act == "mish":
			self.arc = nn.Sequential(				# Placeholder parameter initialization
				nn.Linear(in_out_dim,hidden_dim),	# Mish only uses 1 hidden layer
				nn.Linear(hidden_dim,hidden_dim),
				nn.Linear(hidden_dim,in_out_dim)
				)
			
			if trafo == "sig":
				self.f_fwd = f_sig
				if w_init_:
					self.arc.apply(self.weights_init)

			elif trafo == "log":
				self.f_fwd = f_log
		
		elif act == "ct":

			if n_hidden_l == 1:
				self.arc = nn.Sequential(
					nn.Linear(in_out_dim,hidden_dim),
					mod_CT(grad = True),
					nn.Linear(hidden_dim,hidden_dim),
					mod_CT(grad = True),
					nn.Linear(hidden_dim,in_out_dim)
					)
				if res == False:
					self.f_fwd = f_ct_sig
				else:
					self.f_fwd = f_ct_sig_res
			
			else:	# Extends the network to add more linear layer parameters
				expansion = nn.ModuleList([nn.Linear(in_out_dim,hidden_dim), mod_CT(grad = True)])
				for _ in range(n_hidden_l):
					expansion.extend(nn.ModuleList([nn.Linear(hidden_dim,hidden_dim), mod_CT(grad = True)]))
				compression = nn.ModuleList([nn.Linear(hidden_dim,in_out_dim)])

				self.arc = nn.Sequential(*(expansion + compression))

				self.f_fwd = f_ct_sig_mult_init(n_hidden_l)
			
			if w_init_:
				self.arc.apply(self.weights_init)
		
	def forward(self, x):
		return self.f_fwd(x, self.arc)
	

class DFF_g(nn.Module):
	"""
	Torch module to initialize the parameters for the 
	forward neural network. Kept simpler as training is less complex.

	Args:
		in_out_dim (Int): Dimension of input and ouput. Default: ``1``.
		hidden_dim (Int): Dimension of input and ouput. Default: ``64``.
		trafo (str): Which expansion/compression transformation to use. Default: ``"sig"``.
		act (str): Activation functions between the layers. Default: ``"mish"``.
	"""

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

class weights_init_const:
	"""
	Weight initialization for monotonically increasing transformation.
	"""

	def __init__(self, mean):
		self.mean = mean

	def __call__(self, m):
		if isinstance(m, nn.Linear):
			std = 2/(m.in_features + m.out_features)
			torch.nn.init.normal_(m.weight, mean = self.mean, std=std)


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

def f_ct_sig_res(input, nn_):
	lg, dlg = logit(input)
	l1 = addmm(nn_[0].bias, lg, nn_[0].weight.T)
	ct1, dct1 = ana_CT(l1, nn_[1].beta)
	l2 = addmm(nn_[2].bias, ct1, nn_[2].weight.T)
	ct2, dct2 = ana_CT(l2, nn_[3].beta)
	l3 = addmm(nn_[4].bias, ct2, nn_[4].weight.T)
	sm, dsm = sigmoid(l3 + lg)
	inoutgrad = mul(dlg, dsm)
	jac = matmul(matmul(inoutgrad*dct2*nn_[4].weight, nn_[2].weight)*dct1, nn_[0].weight)
	return sm, jac+inoutgrad

class f_ct_sig_mult_init:
	"""
	Initialization for a network with CT activation functions and
	custom number of hidden layers.
	"""

	def __init__(self, n_hidden_l):
		self.n_hidden_l = n_hidden_l

	def __call__(self, input, nn_):
		return self.f_ct_sig_mult(input, nn_)

	def f_ct_sig_mult(self, input, nn_):

		lg, dlg = logit(input)	# Expansion to reals

		l = addmm(nn_[0].bias, lg, nn_[0].weight.T)	# Input layer
		ct, dct1 = ana_CT(l, nn_[1].beta)

		dct_list = torch.empty((self.n_hidden_l, *dct1.shape), device = l.device)	# Gradient tensor
		
		for i in range(1, self.n_hidden_l+1):	# Run the hidden layers
			l = addmm(nn_[2*i].bias, ct, nn_[2*i].weight.T)
			ct, dct_list[i-1] = ana_CT(l, nn_[2*i+1].beta)

		l = addmm(nn_[-1].bias, ct, nn_[-1].weight.T)	# Final layer

		sm, dsm = sigmoid(l)	# Compression to unit interval

		inoutgrad = mul(dlg, dsm)	# Gradients for logit and sigmoid
		
		jac = nn_[-1].weight	# Jacobian for the last layer
		for i in range(1,self.n_hidden_l+1):   	# Adding Jacobians for the hidden layers
			jac = (dct_list[-i]*jac) @ nn_[-1 - 2*i].weight
		jac = (dct1*jac) @ nn_[0].weight	# Adding Jacobian for the first layer

		return sm, inoutgrad * jac
	

def flown(z1, nn_, base, gamma = 1):
	r"""
	Density transformation with changes of variables.
	Used for likelihood computation or evaluation of 
	the flown density.

	Args:
		z1 (torch.Tensor): Tensor of data at t=1.
		nn_ (pdffit.nn.DFF_f): Backward neural network.
		base (torch.distributions.distribution.Distribution):
			The base distribution (e.g. Beta).
		gamma (float): Exponent for the γ-transformation.

	Returns:
		out (torch.Tensor): Flown data points in log space.
	"""

	if gamma == 1:
		z1_shifted = z1
		gamma_grad = 1.

	else:
		z1_shifted = z1**(1/gamma)
		gamma_grad = (1/gamma) * (z1**(1/gamma - 1))
	z0_out, jac = nn_(z1_shifted)
	jac = jac * gamma_grad
	z0 = torch.clamp(z0_out, min = 1e-24, max = 1-1e-5)
	return base.log_prob(z0).view(-1) + torch.log(torch.clamp(torch.abs(jac), min = 1e-24, max = 1e24).view(-1))

def flown_comb(z1, nn_, base_val, base_sea, scale):
	r"""
	Density transformation with changes of variables.
	Used for likelihood computation or evaluation of 
	the flown density. 
	Uses a composition of a sea-quark-like base density and
	a valence-quark-like density.

	Args:
		z1 (torch.Tensor): Tensor of data at t=1.
		nn_ (pdffit.nn.DFF_f): Backward neural network.
		base_val (torch.distributions.distribution.Distribution):
			The base valence distribution (e.g. Beta).
		base_sea (torch.distributions.distribution.Distribution):
			The base sea distribution (e.g. Beta).
		scale (float): The proprtion between valence and sea base.

	Returns:
		out (torch.Tensor): Flown data points in input space.
	"""

	z0_out, jac = nn_(z1)
	z0 = torch.clamp(z0_out, min = 1e-24, max = 1-1e-5) # Avoid ValueError's 
	jac = torch.log(torch.clamp(torch.abs(jac), min = 1e-24, max = 1e24).view(-1))
	return (scale 	 * (base_val.log_prob(z0).view(-1) + jac).exp() +
		   (1-scale) * (base_sea.log_prob(z0).view(-1) + jac).exp()  )

class flown_comb_init:
	"""
	Initialization for the 
	density transformation with changes of variables.
	Used for likelihood computation or evaluation of 
	the flown density. 
	Uses a composition of a sea-quark-like base density and
	a valence-quark-like density.

	Args:
		z1 (torch.Tensor): Tensor of data/points at t=1.
		base_val (torch.distributions.distribution.Distribution):
			The base valence distribution (e.g. Beta).
		base_sea (torch.distributions.distribution.Distribution):
			The base sea distribution (e.g. Beta).

	Returns:
		out (torch.Tensor): Flown data points in input space.
	"""

	def __init__(self, z1, base_val, base_sea):
		self.z1 = z1 
		self.base_val_lp = base_val.log_prob
		self.base_sea_lp = base_sea.log_prob

	def __call__(self, nn_, scale, jac_out = False):
		r"""
		Args:
			nn_ (pdffit.nn.DFF_f): Backward neural network.
				The base sea distribution (e.g. Beta).
			scale (float): The proprtion between valence and sea base.
			jac_out (Bool): Additionally return the jacobian of the 
				transformation. Default: ``False``.
		"""

		z0_out, jac = nn_(self.z1)
		z0 = torch.clamp(z0_out, min = 1e-24, max = 1-1e-5)	# Avoid ValueError's 
		jac = torch.log(torch.abs(jac).view(-1))
		scale = torch.clamp(scale, min = 0., max = 1.)

		if not jac_out:
			return (scale 	 * (self.base_val_lp(z0).view(-1) + jac).exp() +
				(1-scale) * (self.base_sea_lp(z0).view(-1) + jac).exp()  )
		
		else: 
			return (scale 	 * (self.base_val_lp(z0).view(-1) + jac).exp() +
				(1-scale) * (self.base_sea_lp(z0).view(-1) + jac).exp()  ,
				jac.exp())