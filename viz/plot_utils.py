from . import torch, device, np, p_nlog
import matplotlib.pyplot as plt
from ..nn import DFF_f

plt.rcParams.update({
	"figure.dpi": 100,               
	"savefig.dpi": 100,             
	"figure.figsize": (10, 6),       
})

def plot_densities(f, 
				   g, 
				   p_z0, 
				   p_z1, 
				   fac=1., 
				   target_only = False, 
				   fcolor="r", 
				   tcolor="k", 
				   tlw=4, 
				   talpha=1., 
				   tlabel='Target Beta(1.8, 7.4)', 
				   blabel='Base Beta(2, 7)', 
				   flabel="Flow f(z)", 
				   gamma=1, 
				   bounds = (1e-4, 1-1e-4)
				   ):
	
	z_t1 = torch.linspace(bounds[0],bounds[1],400).reshape(-1,1).to(device)

	if type(fac) == float:
		fac = np.array(fac)

	elif fac == "momentum":
		fac = z_t1.cpu().numpy()

	elif fac == "momentum_u":
		fac = 2*z_t1.cpu().numpy()
			
	# Base density
	if not target_only:
		plt.plot(z_t1.view(-1).cpu().numpy(), 
			fac*np.exp(p_z0.log_prob(z_t1).cpu().numpy()),
			'grey',lw=4,
			label=blabel)

	plt.plot(z_t1.view(-1).cpu().numpy(), 
		fac*np.exp(p_z1.log_prob(z_t1).cpu().numpy()),
		tcolor,lw=tlw,label=tlabel,alpha=talpha)
	
	lg = z_t1.detach().requires_grad_()
	to_base = f(lg)

	if isinstance(f, DFF_f):
		p_theta = -p_nlog(p_z0, *to_base).view(-1)
	else:
		p_theta = p_z0.log_prob(to_base).view(-1) + torch.log(torch.abs(torch.autograd.grad(torch.sum(to_base), lg)[0].view(-1)))

	with torch.no_grad():
		plt.plot(z_t1.view(-1).cpu().numpy(), 
			fac.reshape(-1)*np.exp(p_theta.detach().cpu().numpy()),
			fcolor,lw=1,label=flabel)
	
	if not target_only:
		if isinstance(f, DFF_f):
			inv = g(to_base[0])
		else:
			inv = g(to_base)
		p_phi = p_z0.log_prob(inv).view(-1) + torch.log(torch.abs(torch.autograd.grad(torch.sum(inv), lg)[0].view(-1)))

		with torch.no_grad():
			plt.plot(z_t1.view(-1).cpu().numpy(), 
				fac.reshape(-1)*np.exp(p_phi.detach().cpu().numpy()),
				'b',lw=1, label='Flow + Inverse g(f(z))')
			
	if gamma == 1:	
		pass
	else:
		to_base = f(lg**(1/gamma))
		p_theta_gamma = p_z0.log_prob(to_base).view(-1) + torch.log(torch.abs(torch.autograd.grad(torch.sum(to_base), lg)[0].view(-1)))

		plt.plot(z_t1.view(-1).cpu().numpy(), 
			fac.reshape(-1)*np.exp(p_theta_gamma.detach().cpu().numpy()),
			fcolor, alpha = 0.4, ls = "--", lw=1)


	plt.title("Density Transformation", fontsize=18)
	plt.xlabel("x", fontsize=15)
	plt.ylabel("p(x)", fontsize=15)
	plt.legend(loc='upper right', fontsize=10)


def plot_trafo(f):
	x_range = torch.linspace(1e-4,0.9999,400).reshape(-1,1).to(device)
	plt.plot(x_range.cpu().numpy(), f(x_range)[0].detach().cpu().numpy())