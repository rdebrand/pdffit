from .analytical import *


def get_batch(p_z1 = torch.distributions.Beta(torch.tensor(2., dtype=torch.float32, device=device),torch.tensor(7., dtype=torch.float32, device=device)), 
			  bs = 2048
			  ):
	return p_z1.sample((bs,1)).requires_grad_().to(device)



def sample_hist(n_bins, bs, p_z1):
	samples = p_z1.sample((bs,1)).to(device)
	noise = torch.distributions.Uniform(0., t_one/n_bins).sample((bs,1)).to(device)

	hist = torch.histc(samples, bins=n_bins, min=0, max=1)
	loc = torch.linspace(0, t_one - t_one/n_bins, n_bins).to(device)
	idx = torch.multinomial(hist, bs, replacement=True).view(bs,1)

	return torch.clamp(loc[idx] + noise, 1e-7, 1-1e-7)