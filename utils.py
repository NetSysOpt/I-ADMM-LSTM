import sys
import torch
import datetime
import numpy as np


class EarlyStopping(object):
    def __init__(self, save_path, patience=10):
        dt = datetime.datetime.now()
        self.filename = save_path
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, model, mode, tol, *args):
        if all(vio <= tol for vio in args):
            if self.best_loss is None:
                self.best_loss = loss
                self.save_checkpoint(model)
                self.counter = 0
            elif mode == 'min':
                if (loss <= self.best_loss):
                    self.save_checkpoint(model)
                    self.best_loss = np.min((loss, self.best_loss))
                    self.counter = 0
                else:
                    self.counter += 1
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            elif mode == 'max':
                if (loss >= self.best_loss):
                    self.save_checkpoint(model)
                    self.best_loss = np.max((loss, self.best_loss))
                    self.counter = 0
                else:
                    self.counter += 1
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

        if self.counter >= self.patience:
            self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.filename))


def obj_fn(x, Q, p):
    return 0.5 * torch.bmm(x.permute(0,2,1), torch.bmm(Q, x)) + torch.bmm(p.permute(0, 2, 1), x)

def ineq_dist(x, G, c):
    return torch.clamp(torch.bmm(G, x) - c, 0)

def eq_dist(x, A, b):
    return torch.abs(b - torch.bmm(A, x))

def lb_dist(x, lb):
    return torch.clamp(lb - x, 0)

def ub_dist(x, ub):
    return torch.clamp(x - ub, 0)

def primal_dual_loss(x, y, z, Q, p, A0):
    primal_residual = torch.linalg.vector_norm((torch.bmm(A0, x)-z), dim=(1,2), keepdim=True)
    dual_residual = torch.linalg.vector_norm(torch.bmm(Q, x)+p+torch.bmm(A0.permute(0,2,1), y), dim=(1,2), keepdim=True)
    return primal_residual, dual_residual, primal_residual+dual_residual


def aug_lagr(x, z, y, Q, p, A0, rho_vec):
    fx = 0.5*torch.bmm(x.permute(0,2,1), torch.bmm(Q, p))+torch.bmm(p.permute(0,2,1), x)
    dual_item = torch.bmm(y.permute(0,2,1), torch.bmm(A0, x)-z)
    aug_item = 0.5*(torch.bmm((torch.bmm(A0, x)-z).permute(0,2,1), torch.bmm(torch.diag_embed(rho_vec.squeeze(-1)), torch.bmm(A0, x)-z)))
    return fx+dual_item+aug_item
