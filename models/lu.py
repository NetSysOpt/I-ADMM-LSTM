import torch
import torch.nn as nn

class LU(nn.Module):
    def __init__(self,
                 device):
        super(LU, self).__init__()
        self.device = device

    def name(self):
        return 'torch_solver'

    def forward(self, rho_vec, x, y, z, xv, sigma, A_tild, lu, piv, **kwargs):
        """
        X: [batch_size, num_var, 1]
        """
        # optimizee parameters
        Q = kwargs['Q']
        p = kwargs['p']
        A0 = kwargs['A0']
        zl = kwargs['zl']
        zu = kwargs['zu']

        alpha = 1.6

        # torch.linalg.lu
        if (lu is None) and (piv is None):
            A_tild = torch.concat((torch.concat((Q + sigma * torch.diag_embed(torch.ones(size=(p.shape[0], p.shape[1]), device=p.device)), A0.permute(0, 2, 1)), dim=2),
                                   torch.concat((A0, -(1 / (rho_vec)) * torch.diag_embed(torch.ones(size=(A0.shape[0], A0.shape[1]), device=Q.device))), dim=2)), dim=1)
            b_tild = torch.concat((sigma * x - p, z - (1 / (rho_vec)) * y), dim=1)
            lu, piv = torch.lu(A_tild, pivot=True)
            xv = torch.lu_solve(b_tild, lu, piv)
        else:
            b_tild = torch.concat((sigma * x - p, z - (1 / (rho_vec)) * y), dim=1)
            xv = torch.lu_solve(b_tild, lu, piv)


        x_tild = xv[:, :x.shape[1], :]
        v = xv[:, x.shape[1]:, :]

        z_tild = z + (1/(rho_vec))*(v-y)
        x = alpha * x_tild + (1 - alpha) * x
        z_temp = alpha*z_tild + (1-alpha)*z
        z = torch.max(torch.min(z_temp + (1/(rho_vec))*y, zu), zl)
        y = y+rho_vec*(z_temp-z)

        return x, y, z, xv, A_tild, b_tild, lu, piv