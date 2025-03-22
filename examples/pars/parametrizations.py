import torch
from torch.distributions import Beta as beta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__" and '__file__' not in globals():
    from scipy.integrate import quad
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 100,               
        "savefig.dpi": 100,           
        "figure.figsize": (10, 6),      
    })


def b(a, mean): return a/mean - a
def a(b, mean): return b/(1/mean - 1)


class gluons:
    def __init__(self, p_v, p_s):
        self.p_v = p_v
        self.p_s = p_s
    
    def sample(self, size):
        bs, b = size
        bs = torch.tensor(bs/2)
        return torch.cat((self.p_v.sample((torch.floor(bs).to(torch.int),b)), self.p_s.sample((torch.ceil(bs).to(torch.int),b))), dim=0)
    
    def log_prob(self, x):
        return torch.log(torch.tensor(1/2)*(self.p_v.log_prob(x).exp() + self.p_s.log_prob(x).exp()))


a_g1  = torch.tensor(0.5, device=device)
a_g2 = torch.tensor(-0.5, device=device)
a_q = torch.tensor(-0.5, device=device)

K_u = torch.tensor(3.7, device=device)
K_d = torch.tensor(3.7, device=device)
K_g = torch.tensor(5.0, device=device)
K_q = torch.tensor(6.0, device=device)

delta_uv = torch.tensor(0.228, device=device)
delta_dv = torch.tensor(0.104, device=device)
delta_g1 = torch.tensor(0.249, device=device)
delta_g2 = torch.tensor(0.249, device=device)
delta_usea = torch.tensor(0.104, device=device)/2
delta_dsea = torch.tensor(0.052, device=device)/2
delta_s = torch.tensor(0.010, device=device)/2
delta_c = torch.tensor(0.005, device=device)/2
delta_b = torch.tensor(0.0005, device=device)/2

delta_g = torch.tensor(0.249, device=device) + torch.tensor(0.249, device=device)


b_u  = K_u + 1
b_d = K_d + 1
b_g1 = K_g + 1
b_q = K_q + 1

a_u = a(b_u, delta_uv/2)
a_d = a(b_d, delta_dv)


p_z1_uv = beta(a_u, b_u)
p_z1_dv = beta(a_d, b_d)

p_z1_x_us = beta(a_q+1, b_q)
p_z1_x_ds = beta(a_q+1, b_q)

p_z1_x_g1 = beta(a_g1+1, b_g1)
p_z1_x_g2 = beta(a_g2+1, b_q)


a_u_b = torch.tensor(0.56, device=device)
b_u_b = torch.tensor(4.9, device=device)
a_d_b = torch.tensor(0.57, device=device)
b_d_b = torch.tensor(4.2, device=device)
a_q_b = torch.tensor(-0.4, device=device)
b_q_b = torch.tensor(7.4, device=device)
a_g1_b = torch.tensor(-0.2, device=device)
b_g1_b = torch.tensor(5.8, device=device)
a_g2_b = torch.tensor(-0.45, device=device)


p_z0_uv = beta(a_u_b, b_u_b)
p_z0_dv = beta(a_d_b, b_d_b)

p_z0_x_us = beta(a_q_b+1, b_q_b)
p_z0_x_ds = beta(a_q_b+1, b_q_b)

p_z0_x_g = beta(a_g1_b+1, b_g1_b)

test_p_z0 = beta(torch.tensor(2., device=device), torch.tensor(7., device=device))
test_p_z1 = beta(torch.tensor(1.8, device=device), torch.tensor(7.4, device=device))


if __name__ == "__main__":
    x_range = torch.linspace(1e-4, 1., 500).to(device)

    plt.plot(x_range.cpu().numpy(), (x_range*2*p_z1_uv.log_prob(x_range).exp()).cpu().numpy(), "C0", label='x u_v(x)')
    plt.plot(x_range.cpu().numpy(), (x_range*p_z1_dv.log_prob(x_range).exp()).cpu().numpy(),"C1", label='x d_v(x)')

    plt.plot(x_range.cpu().numpy(), (delta_usea*p_z1_x_us.log_prob(x_range).exp()).cpu().numpy(),"C2", label='x u_s(x)')
    plt.plot(x_range.cpu().numpy(), (delta_dsea*p_z1_x_ds.log_prob(x_range).exp()).cpu().numpy(),"C3", label='x d_s(x)')
    y_g = delta_g1*p_z1_x_g1.log_prob(x_range).exp() + delta_g2*p_z1_x_g2.log_prob(x_range).exp()
    plt.plot(x_range.cpu().numpy(), y_g.cpu().numpy(),"C4", label='x (g_v(x)+g_s(x))')

    plt.plot(x_range.cpu().numpy(), (x_range*2*p_z0_uv.log_prob(x_range).exp()).cpu().numpy(), "C0", alpha = 0.5, ls = "--")
    plt.plot(x_range.cpu().numpy(), (x_range*p_z0_dv.log_prob(x_range).exp()).cpu().numpy(),"C1", alpha = 0.5, ls = "--")

    plt.plot(x_range.cpu().numpy(), (delta_usea*p_z0_x_us.log_prob(x_range).exp()).cpu().numpy(),"C2", alpha = 0.5, ls = "--")
    plt.plot(x_range.cpu().numpy(), (delta_dsea*p_z0_x_ds.log_prob(x_range).exp()).cpu().numpy(),"C3", alpha = 0.5, ls = "--")
    y_g = 2*delta_g1*p_z0_x_g.log_prob(x_range).exp()
    plt.plot(x_range.cpu().numpy(), y_g.cpu().numpy(),"C4", alpha = 0.5, ls = "--")
    plt.plot(0,0, "k", label="Target Density")
    plt.plot(0,0, "k", ls="--", label="Base Density")

    plt.title('Initial Parton Momentum Distributions', fontsize=18)
    plt.ylabel('x f(x)', fontsize=15)
    plt.xlabel('x', fontsize=15)
    plt.xlim(0.,1.)
    plt.ylim(0.,.7)
    plt.legend(fontsize=10)

    plt.savefig("init_pmd.pdf", bbox_inches='tight')