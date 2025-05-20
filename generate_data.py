import os
import sys
import gzip
import pickle
import osqp
import numpy as np
import configargparse
import scipy.io as sio
import torch

from scipy.sparse import csc_matrix

os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', is_config_file=True, type=str)

#optimizee settings
parser.add_argument('--num_var', type=int, help='Number of decision vars.')
parser.add_argument('--num_eq', type=int, help='Number of equality constraints.')
parser.add_argument('--num_ineq', type=int, help='Number of inequality constraints.')
parser.add_argument('--prob_type', type=str, help='Problem type.')
parser.add_argument('--data_size', type=int, help='The number of all instances.')
parser.add_argument('--ptb_deg', type=float, help='Perturbation degree.')




args, _ = parser.parse_known_args()

if args.prob_type == 'QP_RHS':
    dir_path = './datasets/QP_RHS_{}_{}_{}'.format(args.num_var, args.num_ineq, args.num_eq)
    os.makedirs(dir_path, exist_ok=True)

    Q0 = 0.5 * np.diag(np.random.random(args.num_var))
    p0 = np.random.random(size=(args.num_var, 1))
    A0 = np.random.normal(loc=0, scale=1., size=(args.num_eq, args.num_var))
    b0 = np.random.uniform(-1, 1, size=(args.data_size, args.num_eq, 1))
    G0 = np.random.normal(loc=0, scale=1., size=(args.num_ineq, args.num_var))
    c0 = np.sum(np.abs(G0 @ np.linalg.pinv(A0)), axis=1).reshape((args.num_ineq, 1))
    for i in range(args.data_size):
        A01 = np.concatenate((G0, A0), axis=0)
        zl1 = np.concatenate((-np.inf * np.ones(c0.shape), b0[i, :]), axis=0)
        zu1 = np.concatenate((c0, b0[i, :, :]), axis=0)

        solver = osqp.OSQP()
        solver.setup(P=csc_matrix(Q0) * 2, q=p0, A=csc_matrix(A01),
                     l=zl1, u=zu1, verbose=False, eps_prim_inf=1e-4,
                     eps_dual_inf=1e-4, check_termination=1, adaptive_rho_interval=1)
        results = solver.solve()

        if results.info.status == 'solved':
            data_dict = {'Q': Q0, 'p': p0, 'G': G0, 'c': c0,
                         'A': A0, 'b': b0[i, :], 'A0': A01, 'zl': zl1, 'zu': zu1,
                         'x': results.x, 'y': results.y}

            dict_name = './datasets/QP_RHS_{}_{}_{}/qp_rhs_{}.gz'.format(args.num_var, args.num_ineq, args.num_eq, i)
            with gzip.open(dict_name, 'wb') as f:
                pickle.dump(data_dict, f)
        else:
            print('Batch {} optimization failed.'.format(i))

elif args.prob_type == 'QP':
    dir_path = './datasets/QP_{}_{}_{}'.format(args.num_var, args.num_ineq, args.num_eq)
    os.makedirs(dir_path, exist_ok=True)

    Q0 = 0.5 * torch.diag_embed(torch.rand(size=(args.data_size, args.num_var))).numpy()
    p0 = torch.rand(size=(args.data_size, args.num_var)).unsqueeze(-1).numpy()
    A0 = torch.normal(mean=0, std=1, size=(args.data_size, args.num_eq, args.num_var)).numpy()
    b0 = (2 * torch.rand(size=(args.data_size, args.num_eq)).unsqueeze(-1) - 1).numpy()
    G0 = torch.normal(mean=0, std=1, size=(args.data_size, args.num_ineq, args.num_var)).numpy()
    c0 = torch.sum(torch.abs(torch.bmm(torch.tensor(G0), torch.pinverse(torch.tensor(A0)))), dim=2).unsqueeze(-1).numpy()
    for i in range(args.data_size):
        A01 = np.concatenate((G0[i, :, :], A0[i, :, :]), axis=0)
        zl1 = np.concatenate((-np.inf * np.ones(c0[i, :, :].shape), b0[i, :, :]), axis=0)
        zu1 = np.concatenate((c0[i, :, :], b0[i, :, :]), axis=0)

        solver = osqp.OSQP()
        solver.setup(P=csc_matrix(Q0[i, :, :]) * 2, q=p0[i, :, :], A=csc_matrix(A01),
                     l=zl1, u=zu1, verbose=False, eps_prim_inf=1e-4,
                     eps_dual_inf=1e-4, eps_abs=1e-4, eps_rel=1e-4,
                     check_termination=1, adaptive_rho_interval=1, max_iter=20000)
        results = solver.solve()

        if results.info.status == 'solved':
            data_dict = {'Q': Q0[i, :, :], 'p': p0[i, :, :], 'G': G0[i, :, :], 'c': c0[i, :, :],
                         'A': A0[i, :, :], 'b': b0[i, :, :], 'A0': A01, 'zl': zl1, 'zu': zu1,
                         'x': results.x, 'y': results.y}

            dict_name = './datasets/QP_{}_{}_{}/qp_{}.gz'.format(args.num_var, args.num_ineq, args.num_eq, i)
            with gzip.open(dict_name, 'wb') as f:
                pickle.dump(data_dict, f)
        else:
            print('Batch {} optimization failed.'.format(i))

elif args.prob_type == 'Random_QP':
    dir_path = './datasets/Random_QP_{}_{}'.format(args.num_var, args.num_ineq)
    os.makedirs(dir_path, exist_ok=True)
    sparsity = 0.6

    for i in range(args.data_size):
        M = np.random.randn(args.num_var, args.num_var)
        M_mask = np.random.rand(args.num_var, args.num_var) < sparsity
        M = M * M_mask
        Q = (M @ M.T + 0.01 * np.eye(args.num_var)) * 0.5

        A0 = np.random.randn(args.num_ineq, args.num_var)
        A0_mask = np.random.rand(args.num_ineq, args.num_var) < sparsity
        A0 = A0 * A0_mask

        p = np.random.randn(args.num_var, 1)
        zl = -np.random.rand(args.num_ineq, 1)
        zu = np.random.rand(args.num_ineq, 1)

        G = np.concatenate((A0, -A0), axis=0)
        c = np.concatenate((zu, -zl), axis=0)

        solver = osqp.OSQP()
        solver.setup(P=csc_matrix(Q) * 2, q=p, A=csc_matrix(A0),
                     l=zl, u=zu, verbose=False, eps_prim_inf=1e-4,
                     eps_dual_inf=1e-4, eps_abs=1e-4, eps_rel=1e-4,
                     check_termination=1, adaptive_rho_interval=1, max_iter=20000)
        results = solver.solve()

        if results.info.status == 'solved':
            data_dict = {'Q': csc_matrix(Q), 'p': csc_matrix(p), 'G': csc_matrix(G),
                         'c': csc_matrix(c), 'A0': csc_matrix(A0), 'zl': csc_matrix(zl),
                         'zu': csc_matrix(zu)}

            dict_name = './datasets/Random_QP_{}_{}/random_qp_{}.gz'.format(args.num_var, args.num_ineq, i)
            with gzip.open(dict_name, 'wb') as f:
                pickle.dump(data_dict, f)
        else:
            print('Batch {} optimization failed.'.format(i))

elif args.prob_type == 'Equality_QP':
    dir_path = './datasets/Equality_QP_{}_{}'.format(args.num_var, args.num_eq)
    os.makedirs(dir_path, exist_ok=True)
    sparsity = 0.5

    for i in range(args.data_size):
        M = np.random.randn(args.num_var, args.num_var)
        M_mask = np.random.rand(args.num_var, args.num_var) < sparsity
        M = M * M_mask
        Q = (M @ M.T + 0.01 * np.eye(args.num_var)) * 0.5

        p = np.random.randn(args.num_var, 1)

        A = np.random.randn(args.num_eq, args.num_var)
        A_mask = np.random.rand(args.num_eq, args.num_var) < sparsity
        A = A * A_mask

        b = np.random.randn(args.num_eq, 1)

        A0 = A
        zl = b
        zu = b

        solver = osqp.OSQP()
        solver.setup(P=csc_matrix(Q) * 2, q=p, A=csc_matrix(A0),
                     l=zl, u=zu, verbose=False, eps_prim_inf=1e-4,
                     eps_dual_inf=1e-4, eps_abs=1e-4, eps_rel=1e-4,
                     check_termination=1, adaptive_rho_interval=1, max_iter=20000)
        results = solver.solve()

        if results.info.status == 'solved':
            data_dict = {'Q': csc_matrix(Q), 'p': csc_matrix(p), 'A': csc_matrix(A),
                         'b': csc_matrix(b), 'A0': csc_matrix(A0), 'zl': csc_matrix(zl),
                         'zu': csc_matrix(zu)}

            dict_name = './datasets/Equality_QP_{}_{}/equality_qp_{}.gz'.format(args.num_var, args.num_eq, i)
            with gzip.open(dict_name, 'wb') as f:
                pickle.dump(data_dict, f)
        else:
            print('Batch {} optimization failed.'.format(i))

elif args.prob_type == 'SVM':
    dir_path = './datasets/SVM_{}_{}'.format(args.num_var + args.num_ineq, args.num_ineq)
    os.makedirs(dir_path, exist_ok=True)
    total_ites = 50000
    num_solved = 0
    sparsity = 0.5

    for i in range(total_ites):
        I = np.eye(args.num_var)
        Q = np.zeros((args.num_var + args.num_ineq, args.num_var + args.num_ineq))
        Q[:args.num_var, :args.num_var] = I

        lamb = np.random.normal(1)
        p = np.concatenate((np.zeros((args.num_var, 1)), lamb * np.ones((args.num_ineq, 1))), axis=0)

        b_hat = np.concatenate((np.ones(int(args.num_ineq / 2)), -1 * np.ones(int(args.num_ineq / 2))))
        A_hat = np.concatenate((np.random.normal(loc=1 / args.num_var, scale=1 / args.num_var, size=(int(args.num_ineq / 2), args.num_var)),
                                np.random.normal(loc=-1 / args.num_var, scale=1 / args.num_var,
                                                 size=(int(args.num_ineq / 2), args.num_var))), axis=0)
        A_hat_mask = np.random.rand(args.num_ineq, args.num_var) < sparsity
        A_hat = A_hat * A_hat_mask
        G = np.concatenate((np.diag(b_hat) @ A_hat,
                            -np.eye(args.num_ineq)), axis=1)
        c = -np.ones((args.num_ineq, 1))

        lb = np.concatenate((-np.inf * np.ones((args.num_var, 1)), np.zeros((args.num_ineq, 1))), axis=0)
        ub = np.inf * np.ones((args.num_var + args.num_ineq, 1))

        A0 = np.concatenate((G, np.eye(args.num_var + args.num_ineq)), axis=0)
        zl = np.concatenate((-np.inf * np.ones((args.num_ineq, 1)), lb), axis=0)
        zu = np.concatenate((c, ub), axis=0)

        solver = osqp.OSQP()
        solver.setup(P=csc_matrix(Q) * 2, q=p, A=csc_matrix(A0),
                     l=zl, u=zu, verbose=False, eps_prim_inf=1e-4,
                     eps_dual_inf=1e-4, check_termination=1, adaptive_rho_interval=1)
        results = solver.solve()

        if results.info.status == 'solved':
            data_dict = {'Q': csc_matrix(Q), 'p': csc_matrix(p), 'G': csc_matrix(G), 'c': csc_matrix(c),
                         'lb': csc_matrix(lb), 'ub': csc_matrix(ub), 'A0': csc_matrix(A0), 'zl': csc_matrix(zl),
                         'zu': csc_matrix(zu)}

            dict_name = 'E:/gaoxi/OSQP/OSQP-LSTM/datasets/SVM_{}_{}/svm_{}.gz'.format(args.num_var + args.num_ineq, args.num_ineq,
                                                                                      num_solved)
            with gzip.open(dict_name, 'wb') as f:
                pickle.dump(data_dict, f)
            num_solved += 1
            if num_solved >= args.data_size:
                break
        else:
            print('Batch {} optimization failed.'.format(i))


