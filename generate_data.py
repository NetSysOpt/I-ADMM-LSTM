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

elif args.prob_type == 'MM_MOSARQP2':
    # load path
    data_path = './datasets/raw_data'
    mat_path = os.path.join(data_path, args.prob_type[3:] + '.mat')
    mat = sio.loadmat(mat_path)

    # save path
    dir_path = './datasets/{}'.format(args.prob_type)
    os.makedirs(dir_path, exist_ok=True)

    Q0 = mat['Q'].toarray()
    p0 = mat['c']
    G0 = -mat['A'].toarray()
    c0 = -mat['rl']
    l0 = mat['lb']
    u0 = mat['ub']
    num_var = Q0.shape[0]
    num_ineq = G0.shape[0]

    total_ites = 50000
    num_solved = 0

    for i in range(total_ites):
        # perturb quadratic coefficient matrix
        Q1 = Q0.copy()
        nonzero_id = np.nonzero(Q0)
        nonzero_v = Q0[nonzero_id]
        ptb_v = nonzero_v * (1 + np.random.uniform(-args.ptb_deg, args.ptb_deg, size=nonzero_v.shape))
        Q1[nonzero_id] = ptb_v
        Q1 = ((Q1 + Q1.T) / 2) / 2

        # perturb p
        p1 = p0.copy()
        nonzero_id = np.nonzero(p0)
        nonzero_v = p0[nonzero_id]
        ptb_v = nonzero_v * (1 + np.random.uniform(-args.ptb_deg, args.ptb_deg, size=nonzero_v.shape))
        p1[nonzero_id] = ptb_v

        # perturb G
        G1 = G0.copy()
        nonzero_id = np.nonzero(G0)
        nonzero_v = G0[nonzero_id]
        ptb_v = nonzero_v * (1 + np.random.uniform(-args.ptb_deg, args.ptb_deg, size=nonzero_v.shape))
        G1[nonzero_id] = ptb_v

        # perturb b
        c1 = c0.copy()

        # keep the lower bound unchanged
        lb1 = l0.copy()

        # perturb upper bound
        ub1 = u0.copy()

        A01 = np.concatenate((G1, np.eye(num_var)), axis=0)
        zl1 = np.concatenate((-np.inf * np.ones(c1.shape), lb1), axis=0)
        zu1 = np.concatenate((c1, ub1), axis=0)

        solver = osqp.OSQP()
        solver.setup(P=csc_matrix(Q1) * 2, q=p1, A=csc_matrix(A01),
                     l=zl1, u=zu1, verbose=False, eps_prim_inf=1e-4,
                     eps_dual_inf=1e-4, check_termination=1, max_iter=20000,
                     adaptive_rho_interval=1)
        results = solver.solve()

        if results.info.status == 'solved':
            data_dict = {'Q': csc_matrix(Q1), 'p': csc_matrix(p1), 'G': csc_matrix(G1), 'c': csc_matrix(c1),
                         'lb': csc_matrix(lb1), 'ub': csc_matrix(ub1),
                         'A0': csc_matrix(A01), 'zl': csc_matrix(zl1), 'zu': csc_matrix(zu1)}

            dict_name = './datasets/MM_MOSARQP2/mosarqp2_{}.gz'.format(num_solved)
            with gzip.open(dict_name, 'wb') as f:
                pickle.dump(data_dict, f)
            num_solved += 1
            if num_solved >= args.data_size:
                break
        else:
            print('Batch {} optimization failed.'.format(i))


elif args.prob_type == 'MM_QSCRS8':
    # load path
    data_path = './datasets/raw_data'
    mat_path = os.path.join(data_path, args.prob_type[3:] + '.mat')
    mat = sio.loadmat(mat_path)

    # save path
    dir_path = './datasets/{}'.format(args.prob_type)
    os.makedirs(dir_path, exist_ok=True)

    Q0 = mat['Q'].toarray()
    p0 = mat['c']
    G0 = mat['A'].toarray()
    cl0 = mat['rl']
    cu0 = mat['ru']
    l0 = mat['lb']
    u0 = mat['ub']
    num_var = Q0.shape[0]
    num_ineq = G0.shape[0]
    print("The number of variables: {}".format(num_var))
    print("The number of inequality constraints: {}".format(num_ineq * 2))

    total_ites = 50000
    num_solved = 0
    tol = 1e-12

    for i in range(total_ites):
        # perturb quadratic coefficient matrix
        Q1 = Q0.copy()
        nonzero_id = np.nonzero(Q0)
        nonzero_v = Q0[nonzero_id]
        ptb_v = nonzero_v * (1 + np.random.uniform(-args.ptb_deg, args.ptb_deg, size=nonzero_v.shape))
        Q1[nonzero_id] = ptb_v
        Q1 = ((Q1 + Q1.T) / 2) / 2

        # perturb p
        p1 = p0.copy()
        nonzero_id = np.nonzero(p0)
        nonzero_v = p0[nonzero_id]
        ptb_v = nonzero_v * (1 + np.random.uniform(-args.ptb_deg, args.ptb_deg, size=nonzero_v.shape))
        p1[nonzero_id] = ptb_v

        # perturb G
        G1 = G0.copy()
        nonzero_id = np.nonzero(G0)
        nonzero_v = G0[nonzero_id]
        ptb_v = nonzero_v * (1 + np.random.uniform(-args.ptb_deg, args.ptb_deg, size=nonzero_v.shape))
        G1[nonzero_id] = ptb_v

        # perturb cl
        cl1 = cl0.copy()
        non_zero_id = cl0 != 0
        non_inf_id = ~np.isinf(cl0)
        condition = np.logical_and(non_zero_id, non_inf_id)
        non_zero_inf_id = np.nonzero(condition)
        non_zero_inf_v = cl0[non_zero_inf_id]
        ptb_v = non_zero_inf_v * (1 + np.random.uniform(-args.ptb_deg, args.ptb_deg, size=non_zero_inf_v.shape))
        cl1[non_zero_inf_id] = ptb_v
        cl1[np.abs(cu0 - cl0) < tol] = cl0[np.abs(cu0 - cl0) < tol]

        # perturb cu
        cu1 = cu0.copy()
        non_zero_id = cu0 != 0
        non_inf_id = ~np.isinf(cu0)
        condition = np.logical_and(non_zero_id, non_inf_id)
        non_zero_inf_id = np.nonzero(condition)
        non_zero_inf_v = cu0[non_zero_inf_id]
        ptb_v = non_zero_inf_v * (1 + np.random.uniform(-args.ptb_deg, args.ptb_deg, size=non_zero_inf_v.shape))
        cu1[non_zero_inf_id] = ptb_v
        cu1[np.abs(cu0 - cl0) < tol] = cu0[np.abs(cu0 - cl0) < tol]

        # keep the lower bound unchanged
        lb1 = l0.copy()

        # perturb upper bound
        ub1 = u0.copy()

        A01 = np.concatenate((G1, np.eye(num_var)), axis=0)
        zl1 = np.concatenate((cl1, lb1), axis=0)
        zu1 = np.concatenate((cu1, ub1), axis=0)

        G1 = np.concatenate((G1, -G1), axis=0)
        c1 = np.concatenate((cu1, -cl1), axis=0)

        solver = osqp.OSQP()
        solver.setup(P=csc_matrix(Q1) * 2, q=p1, A=csc_matrix(A01),
                     l=zl1, u=zu1, verbose=False, eps_prim_inf=1e-4,
                     eps_dual_inf=1e-4, check_termination=1, max_iter=20000,
                     adaptive_rho_interval=1)
        results = solver.solve()

        if results.info.status == 'solved':
            data_dict = {'Q': csc_matrix(Q1), 'p': csc_matrix(p1), 'G': csc_matrix(G1), 'c': csc_matrix(c1),
                         'lb': csc_matrix(lb1), 'ub': csc_matrix(ub1),
                         'A0': csc_matrix(A01), 'zl': csc_matrix(zl1), 'zu': csc_matrix(zu1)}

            dict_name = './datasets/MM_QSCRS8/qscrs8_{}.gz'.format(num_solved)
            with gzip.open(dict_name, 'wb') as f:
                pickle.dump(data_dict, f)
            num_solved += 1
            if num_solved >= args.data_size:
                break
        else:
            print('Batch {} optimization failed.'.format(i))

elif args.prob_type == 'MM_QSCSD1':
    # load path
    data_path = './datasets/raw_data'
    mat_path = os.path.join(data_path, args.prob_type[3:] + '.mat')
    mat = sio.loadmat(mat_path)

    # save path
    dir_path = './datasets/{}'.format(args.prob_type)
    os.makedirs(dir_path, exist_ok=True)

    Q0 = mat['Q'].toarray()
    p0 = mat['c']
    A0 = mat['A'].toarray()
    b0 = mat['rl']
    l0 = mat['lb']
    u0 = mat['ub']
    num_var = Q0.shape[0]
    num_eq = A0.shape[0]

    total_ites = 50000
    num_solved = 0
    tol = 1e-12

    for i in range(total_ites):
        # perturb quadratic coefficient matrix
        Q1 = Q0.copy()
        nonzero_id = np.nonzero(Q0)
        nonzero_v = Q0[nonzero_id]
        ptb_v = nonzero_v * (1 + np.random.uniform(-args.ptb_deg, args.ptb_deg, size=nonzero_v.shape))
        Q1[nonzero_id] = ptb_v
        Q1 = ((Q1 + Q1.T) / 2) / 2

        # perturb p
        p1 = p0.copy()
        nonzero_id = np.nonzero(p0)
        nonzero_v = p0[nonzero_id]
        ptb_v = nonzero_v * (1 + np.random.uniform(-args.ptb_deg, args.ptb_deg, size=nonzero_v.shape))
        p1[nonzero_id] = ptb_v

        # perturb A
        A1 = A0.copy()

        # perturb b
        b1 = b0.copy()

        # keep the lower bound unchanged
        lb1 = l0.copy()

        # perturb upper bound
        ub1 = u0.copy()

        A01 = np.concatenate((A1, np.eye(num_var)), axis=0)
        zl1 = np.concatenate((b1, lb1), axis=0)
        zu1 = np.concatenate((b1, ub1), axis=0)

        solver = osqp.OSQP()
        solver.setup(P=csc_matrix(Q1) * 2, q=p1, A=csc_matrix(A01),
                     l=zl1, u=zu1, verbose=False, eps_prim_inf=1e-4,
                     eps_dual_inf=1e-4, check_termination=1, max_iter=20000,
                     adaptive_rho_interval=1)
        results = solver.solve()

        if results.info.status == 'solved':
            data_dict = {'Q': csc_matrix(Q1), 'p': csc_matrix(p1), 'A': csc_matrix(A1), 'b': csc_matrix(b1),
                         'lb': csc_matrix(lb1), 'ub': csc_matrix(ub1),
                         'A0': csc_matrix(A01), 'zl': csc_matrix(zl1), 'zu': csc_matrix(zu1)}

            dict_name = './datasets/MM_QSCSD1/qscsd1_{}.gz'.format(num_solved)
            with gzip.open(dict_name, 'wb') as f:
                pickle.dump(data_dict, f)
            num_solved += 1
            if num_solved >= args.data_size:
                break
        else:
            print('Batch {} optimization failed.'.format(i))

elif args.prob_type == 'MM_Q25FV47':
    # load path
    data_path = './datasets/raw_data'
    mat_path = os.path.join(data_path, args.prob_type[3:] + '.mat')
    mat = sio.loadmat(mat_path)

    # save path
    dir_path = './datasets/{}'.format(args.prob_type)
    os.makedirs(dir_path, exist_ok=True)

    Q0 = mat['Q'].toarray()
    p0 = mat['c']
    G0 = mat['A'].toarray()
    cl0 = mat['rl']
    cu0 = mat['ru']
    l0 = mat['lb']
    u0 = mat['ub']
    num_var = Q0.shape[0]
    num_ineq = G0.shape[0]
    print("The number of variables: {}".format(num_var))
    print("The number of inequality constraints: {}".format(num_ineq * 2))

    total_ites = 50000
    num_solved = 0
    tol = 1e-12

    for i in range(total_ites):
        # perturb quadratic coefficient matrix
        Q1 = Q0.copy()
        nonzero_id = np.nonzero(Q0)
        nonzero_v = Q0[nonzero_id]
        ptb_v = nonzero_v * (1 + np.random.uniform(-args.ptb_deg, args.ptb_deg, size=nonzero_v.shape))
        Q1[nonzero_id] = ptb_v
        Q1 = ((Q1 + Q1.T) / 2) / 2

        # perturb p
        p1 = p0.copy()
        nonzero_id = np.nonzero(p0)
        nonzero_v = p0[nonzero_id]
        ptb_v = nonzero_v * (1 + np.random.uniform(-args.ptb_deg, args.ptb_deg, size=nonzero_v.shape))
        p1[nonzero_id] = ptb_v

        # perturb G
        G1 = G0.copy()
        nonzero_id = np.nonzero(G0)
        nonzero_v = G0[nonzero_id]
        ptb_v = nonzero_v * (1 + np.random.uniform(-args.ptb_deg, args.ptb_deg, size=nonzero_v.shape))
        G1[nonzero_id] = ptb_v

        # perturb cl
        cl1 = cl0.copy()
        non_zero_id = cl0 != 0
        non_inf_id = ~np.isinf(cl0)
        condition = np.logical_and(non_zero_id, non_inf_id)
        non_zero_inf_id = np.nonzero(condition)
        non_zero_inf_v = cl0[non_zero_inf_id]
        ptb_v = non_zero_inf_v * (1 + np.random.uniform(-args.ptb_deg, args.ptb_deg, size=non_zero_inf_v.shape))
        cl1[non_zero_inf_id] = ptb_v
        cl1[(cu0 - cl0) < tol] = cl0[(cu0 - cl0) < tol]

        # perturb cu
        cu1 = cu0.copy()
        non_zero_id = cu0 != 0
        non_inf_id = ~np.isinf(cu0)
        condition = np.logical_and(non_zero_id, non_inf_id)
        non_zero_inf_id = np.nonzero(condition)
        non_zero_inf_v = cu0[non_zero_inf_id]
        ptb_v = non_zero_inf_v * (1 + np.random.uniform(-args.ptb_deg, args.ptb_deg, size=non_zero_inf_v.shape))
        cu1[non_zero_inf_id] = ptb_v
        cu1[(cu0 - cl0) < tol] = cu0[(cu0 - cl0) < tol]

        cl2 = cl1.copy()
        cu2 = cu1.copy()
        cl1[(cu2 - cl2) < tol] = cl0[(cu2 - cl2) < tol]
        cu1[(cu2 - cl2) < tol] = cu0[(cu2 - cl2) < tol]

        # keep the lower bound unchanged
        lb1 = l0.copy()

        # perturb upper bound
        ub1 = u0.copy()

        A01 = np.concatenate((G1, np.eye(num_var)), axis=0)
        zl1 = np.concatenate((cl1, lb1), axis=0)
        zu1 = np.concatenate((cu1, ub1), axis=0)

        G1 = np.concatenate((G1, -G1), axis=0)
        c1 = np.concatenate((cu1, -cl1), axis=0)

        solver = osqp.OSQP()
        solver.setup(P=csc_matrix(Q1) * 2, q=p1, A=csc_matrix(A01),
                     l=zl1, u=zu1, verbose=False, eps_prim_inf=1e-4,
                     eps_dual_inf=1e-4, check_termination=1, max_iter=20000,
                     adaptive_rho_interval=1)
        results = solver.solve()

        if results.info.status == 'solved':
            data_dict = {'Q': csc_matrix(Q1), 'p': csc_matrix(p1), 'G': csc_matrix(G1), 'c': csc_matrix(c1),
                         'lb': csc_matrix(lb1), 'ub': csc_matrix(ub1),
                         'A0': csc_matrix(A01), 'zl': csc_matrix(zl1), 'zu': csc_matrix(zu1)}

            dict_name = './datasets/MM_Q25FV47/q25fv47_{}.gz'.format(num_solved)
            with gzip.open(dict_name, 'wb') as f:
                pickle.dump(data_dict, f)
            num_solved += 1
            if num_solved >= args.data_size:
                break
        else:
            print('Batch {} optimization failed.'.format(i))

elif args.prob_type == 'MM_QSHIP04L':
    # load path
    data_path = './datasets/raw_data'
    mat_path = os.path.join(data_path, args.prob_type[3:] + '.mat')
    mat = sio.loadmat(mat_path)

    # save path
    dir_path = './datasets/{}'.format(args.prob_type)
    os.makedirs(dir_path, exist_ok=True)

    Q0 = mat['Q'].toarray()
    p0 = mat['c']
    G0 = mat['A'].toarray()
    cl0 = mat['rl']
    cu0 = mat['ru']
    l0 = mat['lb']
    u0 = mat['ub']
    num_var = Q0.shape[0]
    num_ineq = G0.shape[0]
    print("The number of variables: {}".format(num_var))
    print("The number of inequality constraints: {}".format(num_ineq * 2))

    total_ites = 50000
    num_solved = 0
    tol = 1e-12

    total_ites = 50000
    num_solved = 0
    tol = 1e-12

    for i in range(total_ites):
        # perturb quadratic coefficient matrix
        Q1 = Q0.copy()
        nonzero_id = np.nonzero(Q0)
        nonzero_v = Q0[nonzero_id]
        ptb_v = nonzero_v * (1 + np.random.uniform(-args.ptb_deg, args.ptb_deg, size=nonzero_v.shape))
        Q1[nonzero_id] = ptb_v
        Q1 = ((Q1 + Q1.T) / 2) / 2

        # perturb p
        p1 = p0.copy()
        nonzero_id = np.nonzero(p0)
        nonzero_v = p0[nonzero_id]
        ptb_v = nonzero_v * (1 + np.random.uniform(-args.ptb_deg, args.ptb_deg, size=nonzero_v.shape))
        p1[nonzero_id] = ptb_v

        # perturb G
        G1 = G0.copy()
        nonzero_id = np.nonzero(G0)
        nonzero_v = G0[nonzero_id]
        ptb_v = nonzero_v * (1 + np.random.uniform(-args.ptb_deg, args.ptb_deg, size=nonzero_v.shape))
        G1[nonzero_id] = ptb_v

        # perturb cl
        cl1 = cl0.copy()
        non_zero_id = cl0 != 0
        non_inf_id = ~np.isinf(cl0)
        condition = np.logical_and(non_zero_id, non_inf_id)
        non_zero_inf_id = np.nonzero(condition)
        non_zero_inf_v = cl0[non_zero_inf_id]
        ptb_v = non_zero_inf_v * (1 + np.random.uniform(-args.ptb_deg, args.ptb_deg, size=non_zero_inf_v.shape))
        cl1[non_zero_inf_id] = ptb_v
        cl1[(cu0 - cl0) < tol] = cl0[(cu0 - cl0) < tol]

        # perturb cu
        cu1 = cu0.copy()
        non_zero_id = cu0 != 0
        non_inf_id = ~np.isinf(cu0)
        condition = np.logical_and(non_zero_id, non_inf_id)
        non_zero_inf_id = np.nonzero(condition)
        non_zero_inf_v = cu0[non_zero_inf_id]
        ptb_v = non_zero_inf_v * (1 + np.random.uniform(-args.ptb_deg, args.ptb_deg, size=non_zero_inf_v.shape))
        cu1[non_zero_inf_id] = ptb_v
        cu1[(cu0 - cl0) < tol] = cu0[(cu0 - cl0) < tol]

        cl2 = cl1.copy()
        cu2 = cu1.copy()
        cl1[(cu2 - cl2) < tol] = cl0[(cu2 - cl2) < tol]
        cu1[(cu2 - cl2) < tol] = cu0[(cu2 - cl2) < tol]

        # keep the lower bound unchanged
        lb1 = l0.copy()

        # perturb upper bound
        ub1 = u0.copy()

        A01 = np.concatenate((G1, np.eye(num_var)), axis=0)
        zl1 = np.concatenate((cl1, lb1), axis=0)
        zu1 = np.concatenate((cu1, ub1), axis=0)

        G1 = np.concatenate((G1, -G1), axis=0)
        c1 = np.concatenate((cu1, -cl1), axis=0)

        solver = osqp.OSQP()
        solver.setup(P=csc_matrix(Q1) * 2, q=p1, A=csc_matrix(A01),
                     l=zl1, u=zu1, verbose=False, eps_prim_inf=1e-4,
                     eps_dual_inf=1e-4, check_termination=1, max_iter=20000,
                     adaptive_rho_interval=1)
        results = solver.solve()

        if results.info.status == 'solved':
            data_dict = {'Q': csc_matrix(Q1), 'p': csc_matrix(p1), 'G': csc_matrix(G1), 'c': csc_matrix(c1),
                         'lb': csc_matrix(lb1), 'ub': csc_matrix(ub1),
                         'A0': csc_matrix(A01), 'zl': csc_matrix(zl1), 'zu': csc_matrix(zu1)}

            dict_name = './datasets/MM_QSHIP04L/qship04l_{}.gz'.format(num_solved)
            with gzip.open(dict_name, 'wb') as f:
                pickle.dump(data_dict, f)
            num_solved += 1
            if num_solved >= args.data_size:
                break
        else:
            print('Batch {} optimization failed.'.format(i))

elif args.prob_type == 'MM_QSHIP08S':
    # load path
    data_path = './datasets/raw_data'
    mat_path = os.path.join(data_path, args.prob_type[3:] + '.mat')
    mat = sio.loadmat(mat_path)

    # save path
    dir_path = './datasets/{}'.format(args.prob_type)
    os.makedirs(dir_path, exist_ok=True)

    Q0 = mat['Q'].toarray()
    p0 = mat['c']
    G0 = mat['A'].toarray()
    cl0 = mat['rl']
    cu0 = mat['ru']
    l0 = mat['lb']
    u0 = mat['ub']
    num_var = Q0.shape[0]
    num_ineq = G0.shape[0]
    print("The number of variables: {}".format(num_var))
    print("The number of inequality constraints: {}".format(num_ineq * 2))

    total_ites = 50000
    num_solved = 0
    tol = 1e-12

    for i in range(total_ites):
        # perturb quadratic coefficient matrix
        Q1 = Q0.copy()
        nonzero_id = np.nonzero(Q0)
        nonzero_v = Q0[nonzero_id]
        ptb_v = nonzero_v * (1 + np.random.uniform(-args.ptb_deg, args.ptb_deg, size=nonzero_v.shape))
        Q1[nonzero_id] = ptb_v
        Q1 = ((Q1 + Q1.T) / 2) / 2

        # perturb p
        p1 = p0.copy()
        nonzero_id = np.nonzero(p0)
        nonzero_v = p0[nonzero_id]
        ptb_v = nonzero_v * (1 + np.random.uniform(-args.ptb_deg, args.ptb_deg, size=nonzero_v.shape))
        p1[nonzero_id] = ptb_v

        # perturb G
        G1 = G0.copy()
        nonzero_id = np.nonzero(G0)
        nonzero_v = G0[nonzero_id]
        ptb_v = nonzero_v * (1 + np.random.uniform(-args.ptb_deg, args.ptb_deg, size=nonzero_v.shape))
        G1[nonzero_id] = ptb_v

        # perturb cl
        cl1 = cl0.copy()
        non_zero_id = cl0 != 0
        non_inf_id = ~np.isinf(cl0)
        condition = np.logical_and(non_zero_id, non_inf_id)
        non_zero_inf_id = np.nonzero(condition)
        non_zero_inf_v = cl0[non_zero_inf_id]
        ptb_v = non_zero_inf_v * (1 + np.random.uniform(-args.ptb_deg, args.ptb_deg, size=non_zero_inf_v.shape))
        cl1[non_zero_inf_id] = ptb_v
        cl1[(cu0 - cl0) < tol] = cl0[(cu0 - cl0) < tol]

        # perturb cu
        cu1 = cu0.copy()
        non_zero_id = cu0 != 0
        non_inf_id = ~np.isinf(cu0)
        condition = np.logical_and(non_zero_id, non_inf_id)
        non_zero_inf_id = np.nonzero(condition)
        non_zero_inf_v = cu0[non_zero_inf_id]
        ptb_v = non_zero_inf_v * (1 + np.random.uniform(-args.ptb_deg, args.ptb_deg, size=non_zero_inf_v.shape))
        cu1[non_zero_inf_id] = ptb_v
        cu1[(cu0 - cl0) < tol] = cu0[(cu0 - cl0) < tol]

        cl2 = cl1.copy()
        cu2 = cu1.copy()
        cl1[(cu2 - cl2) < tol] = cl0[(cu2 - cl2) < tol]
        cu1[(cu2 - cl2) < tol] = cu0[(cu2 - cl2) < tol]

        # keep the lower bound unchanged
        lb1 = l0.copy()

        # perturb upper bound
        ub1 = u0.copy()

        A01 = np.concatenate((G1, np.eye(num_var)), axis=0)
        zl1 = np.concatenate((cl1, lb1), axis=0)
        zu1 = np.concatenate((cu1, ub1), axis=0)

        G1 = np.concatenate((G1, -G1), axis=0)
        c1 = np.concatenate((cu1, -cl1), axis=0)

        solver = osqp.OSQP()
        solver.setup(P=csc_matrix(Q1) * 2, q=p1, A=csc_matrix(A01),
                     l=zl1, u=zu1, verbose=False, eps_prim_inf=1e-4,
                     eps_dual_inf=1e-4, check_termination=1, max_iter=20000,
                     adaptive_rho_interval=1)
        results = solver.solve()

        if results.info.status == 'solved':
            data_dict = {'Q': csc_matrix(Q1), 'p': csc_matrix(p1), 'G': csc_matrix(G1), 'c': csc_matrix(c1),
                         'lb': csc_matrix(lb1), 'ub': csc_matrix(ub1),
                         'A0': csc_matrix(A01), 'zl': csc_matrix(zl1), 'zu': csc_matrix(zu1)}

            dict_name = './datasets/MM_QSHIP08S/qship08s_{}.gz'.format(num_solved)
            with gzip.open(dict_name, 'wb') as f:
                pickle.dump(data_dict, f)
            num_solved += 1
            if num_solved >= args.data_size:
                break
        else:
            print('Batch {} optimization failed.'.format(i))


elif args.prob_type == 'MM_CVXQP1_M':
    # load path
    data_path = './datasets/raw_data'
    mat_path = os.path.join(data_path, args.prob_type[3:] + '.mat')
    mat = sio.loadmat(mat_path)

    # save path
    dir_path = './datasets/{}'.format(args.prob_type)
    os.makedirs(dir_path, exist_ok=True)

    Q0 = mat['Q'].toarray()
    p0 = mat['c']
    A0 = mat['A'].toarray()
    b0 = mat['rl']
    l0 = mat['lb']
    u0 = mat['ub']
    num_var = Q0.shape[0]
    num_eq = A0.shape[0]
    print("The number of variables: {}".format(num_var))
    print("The number of equality constraints: {}".format(num_eq))

    total_ites = 50000
    num_samples = 1000
    num_solved = 0

    for i in range(total_ites):
        Q1 = Q0.copy()

        # perturb p
        p1 = p0.copy()
        nonzero_id = np.nonzero(p0)
        nonzero_v = p0[nonzero_id]
        ptb_v = nonzero_v * (1 + np.random.uniform(-args.ptb_deg, args.ptb_deg, size=nonzero_v.shape))
        p1[nonzero_id] = ptb_v

        # perturb A
        A1 = A0.copy()

        # perturb b
        b1 = b0.copy()

        # perturb lb
        lb1 = l0.copy()
        nonzero_id = np.nonzero(l0)
        nonzero_v = l0[nonzero_id]
        ptb_v = nonzero_v * (1 + np.random.uniform(-args.ptb_deg, args.ptb_deg, size=nonzero_v.shape))
        lb1[nonzero_id] = ptb_v

        # perturb upper bound
        ub1 = u0.copy()
        nonzero_id = np.nonzero(u0)
        nonzero_v = u0[nonzero_id]
        ptb_v = nonzero_v * (1 + np.random.uniform(-args.ptb_deg, args.ptb_deg, size=nonzero_v.shape))
        ub1[nonzero_id] = ptb_v

        A01 = np.concatenate((A1, np.eye(num_var)), axis=0)
        zl1 = np.concatenate((b1, lb1), axis=0)
        zu1 = np.concatenate((b1, ub1), axis=0)

        solver = osqp.OSQP()
        solver.setup(P=csc_matrix(Q1) * 2, q=p1, A=csc_matrix(A01),
                     l=zl1, u=zu1, verbose=False, eps_prim_inf=1e-4,
                     eps_dual_inf=1e-4, check_termination=1, max_iter=20000,
                     adaptive_rho_interval=1)
        results = solver.solve()

        if results.info.status == 'solved':
            data_dict = {'Q': csc_matrix(Q1), 'p': csc_matrix(p1), 'A': csc_matrix(A1), 'b': csc_matrix(b1),
                         'lb': csc_matrix(lb1), 'ub': csc_matrix(ub1),
                         'A0': csc_matrix(A01), 'zl': csc_matrix(zl1), 'zu': csc_matrix(zu1)}

            dict_name = './datasets/MM_CVXQP1_M/cvxqp1_m_{}.gz'.format(num_solved)
            with gzip.open(dict_name, 'wb') as f:
                pickle.dump(data_dict, f)
            num_solved += 1
            if num_solved >= num_samples:
                break
        else:
            print('Batch {} optimization failed.'.format(i))

elif args.prob_type == 'MM_CVXQP1_M':
    # load path
    data_path = './datasets/raw_data'
    mat_path = os.path.join(data_path, args.prob_type[3:] + '.mat')
    mat = sio.loadmat(mat_path)

    # save path
    dir_path = './datasets/{}'.format(args.prob_type)
    os.makedirs(dir_path, exist_ok=True)

    Q0 = mat['Q'].toarray()
    p0 = mat['c']
    A0 = mat['A'].toarray()
    b0 = mat['rl']
    l0 = mat['lb']
    u0 = mat['ub']
    num_var = Q0.shape[0]
    num_eq = A0.shape[0]
    print("The number of variables: {}".format(num_var))
    print("The number of equality constraints: {}".format(num_eq))

    total_ites = 50000
    num_solved = 0

    for i in range(total_ites):
        # perturb quadratic coefficient matrix
        Q1 = Q0.copy()
        Q1 = ((Q1 + Q1.T) / 2) / 2

        # perturb p
        p1 = p0.copy()
        nonzero_id = np.nonzero(p0)
        nonzero_v = p0[nonzero_id]
        ptb_v = nonzero_v * (1 + np.random.uniform(-args.ptb_deg, args.ptb_deg, size=nonzero_v.shape))
        p1[nonzero_id] = ptb_v

        # perturb A
        A1 = A0.copy()

        # perturb b
        b1 = b0.copy()

        # perturb lb
        lb1 = l0.copy()
        nonzero_id = np.nonzero(l0)
        nonzero_v = l0[nonzero_id]
        ptb_v = nonzero_v * (1 + np.random.uniform(-args.ptb_deg, args.ptb_deg, size=nonzero_v.shape))
        lb1[nonzero_id] = ptb_v

        # perturb upper bound
        ub1 = u0.copy()
        nonzero_id = np.nonzero(u0)
        nonzero_v = u0[nonzero_id]
        ptb_v = nonzero_v * (1 + np.random.uniform(-args.ptb_deg, args.ptb_deg, size=nonzero_v.shape))
        ub1[nonzero_id] = ptb_v

        A01 = np.concatenate((A1, np.eye(num_var)), axis=0)
        zl1 = np.concatenate((b1, lb1), axis=0)
        zu1 = np.concatenate((b1, ub1), axis=0)

        solver = osqp.OSQP()
        solver.setup(P=csc_matrix(Q1) * 2, q=p1, A=csc_matrix(A01),
                     l=zl1, u=zu1, verbose=False, eps_prim_inf=1e-4,
                     eps_dual_inf=1e-4, check_termination=1, max_iter=20000,
                     adaptive_rho_interval=1)
        results = solver.solve()

        if results.info.status == 'solved':
            data_dict = {'Q': csc_matrix(Q1), 'p': csc_matrix(p1), 'A': csc_matrix(A1), 'b': csc_matrix(b1),
                         'lb': csc_matrix(lb1), 'ub': csc_matrix(ub1),
                         'A0': csc_matrix(A01), 'zl': csc_matrix(zl1), 'zu': csc_matrix(zu1)}

            dict_name = './datasets/MM_CVXQP3_M/cvxqp3_m_{}.gz'.format(num_solved)
            with gzip.open(dict_name, 'wb') as f:
                pickle.dump(data_dict, f)
            num_solved += 1
            if num_solved >= args.data_size:
                break
        else:
            print('Batch {} optimization failed.'.format(i))




