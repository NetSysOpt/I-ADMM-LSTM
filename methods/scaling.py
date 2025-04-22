import torch
import sys


class Scaling(object):

    def __init__(self, num_var, num_constr, scaling_ites, device):
        self.n = num_var
        self.m = num_constr
        self.device = device
        self.scaling_ites = scaling_ites
        self.MIN_SCALING = 1e-04
        self.MAX_SCALING = 1e04
        self.D = None
        self.E = None

    def _norm_KKT_cols(self, Q, A0):
        """
        Compute the norm of KKT matrix from Q and A0
        """
        #First half
        norm_Q_cols = torch.linalg.norm(Q, ord=torch.inf, dim=1)
        norm_A0_cols = torch.linalg.norm(A0, ord=torch.inf, dim=1)
        norm_first_half = torch.max(norm_Q_cols, norm_A0_cols)

        #Second half
        norm_second_half = torch.linalg.norm(A0, ord=torch.inf, dim=2)

        return torch.concat((norm_first_half, norm_second_half), dim=-1)

    def _limit_scaling(self, norm_vec):
        """
        Norm vector for scaling
        """
        if isinstance(norm_vec, (list, tuple, torch.Tensor)):
            new_norm_vec = torch.clamp(norm_vec, min=self.MIN_SCALING, max=self.MAX_SCALING)
            new_norm_vec[new_norm_vec==self.MIN_SCALING] = 1.0
        else:
            if norm_vec < self.MIN_SCALING:
                new_norm_vec = 1.0
            elif norm_vec > self.MAX_SCALING:
                new_norm_vec = self.MAX_SCALING
            else:
                new_norm_vec = norm_vec

        return new_norm_vec



    def scale_data(self, Q, p, A0, lb, ub):
        batch_size = Q.shape[0]

        s_temp = torch.ones(self.n+self.m, device=self.device)
        c = 1.0

        # Initialize scaler matrices
        D = torch.diag_embed(torch.ones(size=(batch_size, self.n), device=self.device))
        if self.m == 0:
            E = torch.tensor([], device=self.device)
        else:
            E = torch.diag_embed(torch.ones(size=(batch_size, self.m), device=self.device))

        #Iterate scaling
        for i in range(self.scaling_ites):
            # First Step Ruiz
            norm_cols = self._norm_KKT_cols(Q, A0)
            norm_cols = self._limit_scaling(norm_cols)  # Limit scaling
            sqrt_norm_cols = torch.sqrt(norm_cols)  # Compute sqrt
            s_temp = torch.reciprocal(sqrt_norm_cols)  # Elementwise recipr

            # Obtain Scalar Matrices
            D_temp = torch.diag_embed(s_temp[:, :self.n])

            if self.m == 0:
                E_temp = 0
            else:
                E_temp = torch.diag_embed(s_temp[:, self.n:])

            # Scale data in place
            Q = torch.bmm(D_temp, torch.bmm(Q, D_temp))
            A0 = torch.bmm(E_temp, torch.bmm(A0, D_temp))
            p = torch.bmm(D_temp, p)
            lb = E_temp.diagonal(dim1=1, dim2=2).unsqueeze(-1)*lb
            ub = E_temp.diagonal(dim1=1, dim2=2).unsqueeze(-1)*ub

            # Updata equilibration matrices D and E
            D = torch.bmm(D_temp, D)
            E = torch.bmm(E_temp, E)

            # Second Step cost normalization
            norm_Q_cols = torch.linalg.norm(Q, ord=torch.inf, dim=1).mean(-1, keepdim=True)
            inf_norm_p = torch.linalg.norm(p, ord=torch.inf, dim=1)
            inf_norm_p = self._limit_scaling(inf_norm_p)
            scale_cost = torch.max(inf_norm_p, norm_Q_cols)
            scale_cost = self._limit_scaling(scale_cost)
            scale_cost = 1.0 / scale_cost

            c_temp = scale_cost

            # normlization cost
            Q = c_temp.unsqueeze(-1) * Q
            p = c_temp.unsqueeze(-1) * p

            # Update scaling
            c = c_temp.unsqueeze(-1) * c

        self.D = D
        self.D_inv = torch.diag_embed(torch.reciprocal(D.diagonal(dim1=-2, dim2=-1)))

        self.E = E
        if self.m == 0:
            self.Einv = E
        else:
            self.Einv = torch.diag_embed(torch.reciprocal(E.diagonal(dim1=-2, dim2=-1)))

        self.c = c
        self.cinv = 1.0 / c

        return Q, p, A0, lb, ub