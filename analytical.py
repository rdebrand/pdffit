from . import torch, device

t_one = torch.tensor(1., device=device)
t_two = torch.tensor(2., device=device)
ln10 = torch.log(torch.tensor(10., device=device))

sp = torch.nn.Softplus()
th = torch.nn.Tanh()

def mish(l):
	"""
	Mish activation function and gradient.
	"""
	thspm = th(sp(l))
	return torch.mul(l, thspm), torch.addcmul(thspm, l, torch.mul(torch.sigmoid(l), (t_one - torch.pow(thspm, t_two))))

def logit(s):
	"""
	Logit expansion and gradient.
	"""
	lg = torch.logit(s)
	return lg, t_one/(s*(t_one - s))

def sigmoid(l):
	"""
	Sigmoid compression and gradient.
	"""
	sg = torch.sigmoid(l)
	return sg, sg*(t_one - sg)

def log10(s):
	return torch.log10(s), 1/(ln10*s)

def exp10(l):
	pw = torch.pow(10., l)
	return pw, ln10*pw

def ana_CT(l, beta, grad_ = True, coeff = 0.5, threshold = 20.):
	"""
	Curvature tuning function and gradient w.r.t. to input.
	Adaptation from https://arxiv.org/pdf/2502.07783.
	"""
	inv_beta_ = 1/(1-beta)
	l_inv_beta_ = l*inv_beta_
	beta_inv_beta_ = beta*inv_beta_
	log_1_ = torch.log(1 + torch.exp(l_inv_beta_))
	sigm_1_ = torch.sigmoid(l_inv_beta_)
	sigm_beta_ = torch.sigmoid(beta*l_inv_beta_)

	normal_ver = coeff*sigm_beta_*l + (1 - coeff)*log_1_*(1-beta)
	overflow_ver = (coeff*sigm_beta_ + (1 - coeff))*l

	if grad_:	
		term_1 = coeff*sigm_beta_*(1 + (1 - sigm_beta_)*l*beta_inv_beta_)
		overflow_grad_l = term_1 + 1 - coeff
		normal_grad_l = term_1 + (1 - coeff)*sigm_1_
		
		return torch.where(l_inv_beta_ <= threshold, normal_ver, overflow_ver), torch.where(l_inv_beta_ <= threshold, normal_grad_l, overflow_grad_l)
	
	else:

		return torch.where(l_inv_beta_ <= threshold, normal_ver, overflow_ver)