from . import torch, device
import torch.nn as nn


t_one = torch.tensor(1., dtype=torch.float32, device=device)
t_two = torch.tensor(2., dtype=torch.float32, device=device)
t_clamp_max = torch.tensor(1-1e-4, dtype=torch.float32, device=device)


class lgnn_f(torch.autograd.Function):
	def __init__(self):
		super().__init__()
	
	@staticmethod
	def forward(ctx, input):
		ctx.save_for_backward(input)
		return torch.logit(input)

	@staticmethod
	def backward(ctx, grad_output):
		input, = ctx.saved_tensors
		grad_logit = t_one/(input * (t_one - input))
		return grad_logit * grad_output
	

class lgnn(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return lgnn_f.apply(x)


class Log10(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.log10(x)
    
class Exp10(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.pow(10, x)


class clamp_f(torch.autograd.Function):
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def forward(ctx, input):
        return torch.clamp(input,  max = t_clamp_max)

    @staticmethod
    def backward(ctx, grad_output):
        return t_one * grad_output
    
class clamp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return clamp_f.apply(x)