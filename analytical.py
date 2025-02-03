from . import torch, device

t_one = torch.tensor(1., dtype=torch.float32, device=device)
t_two = torch.tensor(2., dtype=torch.float32, device=device)
ln10 = torch.log(torch.tensor(10., dtype=torch.float32, device=device))

sp = torch.nn.Softplus()
th = torch.nn.Tanh()

def mish(l):
	thspm = th(sp(l))
	return torch.mul(l, thspm), torch.addcmul(thspm, l, torch.mul(torch.sigmoid(l), (t_one - torch.pow(thspm, t_two))))

def logit(s):
	lg = torch.logit(s)
	return lg, t_one/(s*(t_one - s))

def sigmoid(l):
	sg = torch.sigmoid(l)
	return sg, sg*(t_one - sg)

def log10(s):
	return torch.log10(s), 1/(ln10*s)

def exp10(l):
	pw = torch.pow(10., l)
	return pw, ln10*pw
