import torch
import sys
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self,
                 num_constr,
                 input_dim,
                 hidden_dim,
                 length,
                 device):
        super(LSTM, self).__init__()
        self.num_constr = num_constr
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.length = length
        self.RHO_EQ_OVER_RHO_INEQ = 1e03
        self.device = device

        self.W_i = nn.Parameter(torch.normal(mean=0, std=0.01, size=(input_dim, hidden_dim), device=self.device), requires_grad=True)
        self.U_i = nn.Parameter(torch.normal(mean=0, std=0.01, size=(hidden_dim, hidden_dim), device=self.device), requires_grad=True)
        self.b_i = nn.Parameter(torch.zeros((hidden_dim), device=device, dtype=torch.float32), requires_grad=True)

        self.W_f = nn.Parameter(torch.normal(mean=0, std=0.01, size=(input_dim, hidden_dim), device=self.device), requires_grad=True)
        self.U_f = nn.Parameter(torch.normal(mean=0, std=0.01, size=(hidden_dim, hidden_dim), device=self.device), requires_grad=True)
        self.b_f = nn.Parameter(torch.zeros((hidden_dim), device=device, dtype=torch.float32), requires_grad=True)

        self.W_o = nn.Parameter(torch.normal(mean=0, std=0.01, size=(input_dim, hidden_dim), device=self.device), requires_grad=True)
        self.U_o = nn.Parameter(torch.normal(mean=0, std=0.01, size=(hidden_dim, hidden_dim), device=self.device), requires_grad=True)
        self.b_o = nn.Parameter(torch.zeros((hidden_dim), device=device, dtype=torch.float32), requires_grad=True)

        self.W_u = nn.Parameter(torch.normal(mean=0, std=0.01, size=(input_dim, hidden_dim), device=self.device), requires_grad=True)
        self.U_u = nn.Parameter(torch.normal(mean=0, std=0.01, size=(hidden_dim, hidden_dim), device=self.device), requires_grad=True)
        self.b_u = nn.Parameter(torch.zeros((hidden_dim), device=device, dtype=torch.float32), requires_grad=True)

        self.W_h = nn.Parameter(torch.normal(mean=0, std=0.01, size=(hidden_dim, 1), device=self.device), requires_grad=True)
        self.b_h = nn.Parameter(torch.zeros((1), device=device, dtype=torch.float32), requires_grad=True)

        self.rho = nn.Parameter(torch.normal(mean=0, std=0.01, size=(length, 1), device=self.device), requires_grad=True)
        self.alpha = nn.Parameter(torch.normal(mean=0, std=0.01, size=(self.length, 1), device=self.device), requires_grad=True)

    def name(self):
        return 'lstm'


    def forward(self, t, num_ineq, num_eq, x, y, z, xv, sigma, H_t, C_t, **kwargs):
        """
        X: [batch_size, num_var, 1]
        """
        # optimizee parameters
        Q = kwargs['Q']
        p = kwargs['p']
        lb = kwargs['lb']
        ub = kwargs['ub']
        A0 = kwargs['A0']
        zl = kwargs['zl']
        zu = kwargs['zu']

        rho = torch.sigmoid(self.rho[t,:])
        rho_vec = torch.ones(size=y.shape, device=self.device)*rho
        rho_vec[:, num_ineq:num_ineq+num_eq, :] = rho_vec[:, num_ineq:num_ineq+num_eq, :]*self.RHO_EQ_OVER_RHO_INEQ
        alpha = 2 * torch.sigmoid(self.alpha[t, :])
        #alpha = 1.6

        # direct
        A_tild = torch.concat((torch.concat((Q + sigma * torch.diag_embed(torch.ones(size=(Q.shape[0], Q.shape[1]), device=Q.device)), A0.permute(0, 2, 1)), dim=2),
                                   torch.concat((A0, -(1 / rho_vec) * torch.diag_embed(torch.ones(size=(A0.shape[0], A0.shape[1]), device=Q.device))), dim=2)), dim=1)
        b_tild = torch.concat((sigma * x - p, z - (1 / rho_vec) * y), dim=1)


        inputs = torch.concat([xv, torch.bmm(A_tild.permute(0,2,1), torch.bmm(A_tild, xv)-b_tild)], dim=-1)
        #inputs = torch.concat([xv, torch.bmm(A_tild, xv) - b_tild, b_tild - torch.bmm(A_tild, xv)], dim=-1)
        I_t = torch.sigmoid(inputs @ self.W_i + H_t @ self.U_i + self.b_i)
        F_t = torch.sigmoid(inputs @ self.W_f + H_t @ self.U_f + self.b_f)
        O_t = torch.sigmoid(inputs @ self.W_o + H_t @ self.U_o + self.b_o)
        U_t = torch.tanh(inputs @ self.W_u + H_t @ self.U_u + self.b_u)
        C_t = I_t * U_t + F_t * C_t
        H_t = O_t * torch.tanh(C_t)
        grad = H_t @ self.W_h + self.b_h

        xv = xv - grad

        x_tild = xv[:, :x.shape[1], :]
        v = xv[:, x.shape[1]:, :]

        z_tild = z + (1/(rho_vec))*(v-y)
        x = alpha * x_tild + (1 - alpha) * x
        # if (lb is not None) and (ub is not None):
        #     x = torch.max(torch.min(x, ub), lb)
        #z_temp = alpha * z_tild + (1 - alpha) * z
        z_temp = z_tild
        z = torch.max(torch.min(z_temp + (1/(rho_vec))*y, zu), zl)
        y = y+rho_vec*(z_temp - z)

        return x, y, z, xv, H_t, C_t, A_tild, b_tild, rho_vec