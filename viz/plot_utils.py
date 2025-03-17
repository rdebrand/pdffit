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
				   tb_disp = (True, True),
				   fi_disp = (True, True),
				   fcolor="r", 
				   icolor="b",
				   tcolor="k",
				   bcolor="grey", 
				   tlw=4, 
				   blw=4,
				   talpha=1., 
				   balpha=1.,
				   tlabel='Target Beta(1.8, 7.4)', 
				   blabel='Base Beta(2, 7)', 
				   flabel="Flow f(z)", 
				   gflabel ="Flow + Inverse g(f(z))",
				   gamma=1, 
				   bounds = (1e-4, 1-1e-4),
				   title_fs = 18, label_fs = 15, legend_fs = 10
				   ):
	
	z_t1 = torch.linspace(bounds[0],bounds[1],400).reshape(-1,1).to(device)

	if type(fac) == float:
		fac = np.array(fac)

	elif fac == "momentum":
		fac = z_t1.cpu().numpy()

	elif fac == "momentum_u":
		fac = 2*z_t1.cpu().numpy()
			
	# Base density
	if tb_disp[1]:
		plt.plot(z_t1.view(-1).cpu().numpy(), 
			fac*np.exp(p_z0.log_prob(z_t1).cpu().numpy()),
			color=bcolor, alpha=balpha,lw=blw,label=blabel)
		
	# Target density
	if tb_disp[0]:
		plt.plot(z_t1.view(-1).cpu().numpy(), 
			fac*np.exp(p_z1.log_prob(z_t1).cpu().numpy()),
			color=tcolor,lw=tlw,label=tlabel,alpha=talpha)
	
	lg = z_t1.detach().requires_grad_()
	to_base = f(lg)

	if isinstance(f, DFF_f):
		p_theta = -p_nlog(p_z0, *to_base).view(-1)
	else:
		p_theta = p_z0.log_prob(to_base).view(-1) + torch.log(torch.abs(torch.autograd.grad(torch.sum(to_base), lg, retain_graph=True)[0].view(-1)))

	with torch.no_grad():
		plt.plot(z_t1.view(-1).cpu().numpy(), 
			fac.reshape(-1)*np.exp(p_theta.detach().cpu().numpy()),
			color=fcolor, lw=1, label=flabel)
	
	if fi_disp[1]:
		if isinstance(f, DFF_f):
			inv = g(to_base[0])
		else:
			inv = g(to_base)
		p_phi = p_z0.log_prob(inv).view(-1) + torch.log(torch.abs(torch.autograd.grad(torch.sum(inv), lg)[0].view(-1))) 

		with torch.no_grad():
			plt.plot(z_t1.view(-1).cpu().numpy(), 
				fac.reshape(-1)*np.exp(p_phi.detach().cpu().numpy()),
				color=icolor, lw=1, label=gflabel)
			
	if gamma == 1:	
		pass
	else:
		to_base = f(lg**(1/gamma))
		if isinstance(f, DFF_f):
			out, jac = to_base
			jac = jac * 1/gamma * lg**(1/gamma-1)
			p_theta_gamma = -p_nlog(p_z0, out, jac).view(-1)
		else:
			p_theta_gamma = p_z0.log_prob(to_base).view(-1) + torch.log(torch.abs(torch.autograd.grad(torch.sum(to_base), lg)[0].view(-1)))

		plt.plot(z_t1.view(-1).cpu().numpy(), 
			fac.reshape(-1)*np.exp(p_theta_gamma.detach().cpu().numpy()),
			fcolor, alpha = 0.4, ls = "--", lw=1)


	plt.title("Density Transformation", fontsize=title_fs)
	plt.xlabel("x", fontsize=label_fs)
	plt.ylabel("p(x)", fontsize=label_fs)
	plt.legend(loc='upper right', fontsize=legend_fs)


def plot_trafo(f):
	x_range = torch.linspace(1e-4,0.9999,400).reshape(-1,1).to(device)
	if isinstance(f, DFF_f):
		plt.plot(x_range.cpu().numpy(), f(x_range)[0].detach().cpu().numpy())
	else:
		plt.plot(x_range.cpu().numpy(), f(x_range).detach().cpu().numpy())