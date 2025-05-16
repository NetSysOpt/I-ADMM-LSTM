import os
import sys
import random
import numpy as np
import configargparse
import time
import gzip
import pickle
import datetime
import torch
import torch.optim as optim
import scipy.io as sio

from methods.scaling import Scaling

from models.lstm import LSTM
from models.lu import LU
from utils import (EarlyStopping, primal_dual_loss, obj_fn, ineq_dist, eq_dist, lb_dist, ub_dist, aug_lagr)



parser = configargparse.ArgumentParser(description='train')
parser.add_argument('-c', '--config', is_config_file=True, type=str)

#optimizee settings
parser.add_argument('--num_var', type=int, help='Number of decision vars.')
parser.add_argument('--num_eq', type=int, help='Number of equality constraints.')
parser.add_argument('--num_ineq', type=int, help='Number of inequality constraints.')
parser.add_argument('--prob_type', type=str, help='Problem type.')
parser.add_argument('--qplib_num', type=int, help='The number of QPLIB instance.')

#model settings
parser.add_argument('--scaling_ites', type=int, default=10, help='Number of scaling iterations.')
parser.add_argument('--input_dim', type=int, default=2, help='Input feature dimensions of deep learning optimizer.')
parser.add_argument('--hidden_dim', type=int, help='Hidden dimensions of deep learning optimizer.')
parser.add_argument('--model_name', type=str, help='The optimizer name.')
parser.add_argument('--num_layer', type=int, help='Number of model layers.')
parser.add_argument('--sigma', type=float, help='Penalty parameter.')

#training settings
parser.add_argument('--eq_tol', type=float, help='Equality tolerance for model saving.')
parser.add_argument('--ineq_tol', type=float, help='Inequality tolerance for model saving.')
parser.add_argument('--truncated_length', type=int, help='Length of TBPTT.')
parser.add_argument('--val_frac', type=float, help='The proportion of validation set to the total size of the dataset.')
parser.add_argument('--test_frac', type=float, help='The proportion of test set to the total size of the dataset.')
parser.add_argument('--batch_size', type=int, help='training batch size.')
parser.add_argument('--device', type=str, help='cuda.')
parser.add_argument('--lr', type=float, help='Learning rate.')
parser.add_argument('--num_epoch', type=int, help='Global training epochs.')
parser.add_argument('--outer_T', type=int, help='Iteration of proposed model.')
parser.add_argument('--early_stop_mode', type=str, help='Min or Max.')
parser.add_argument('--patience', type=int, default=100, help='Patience of model saving.')
parser.add_argument('--save_dir', type=str, default='./results/', help='Save path for the best model.')
parser.add_argument('--save_sol', action='store_true', help='Save the results.')
parser.add_argument('--seed', type=int, default=17, help='random seed.')
parser.add_argument('--scaling', action='store_true', help='Perform symmetric diagonal scaling via equilibration.')
parser.add_argument('--test', action='store_true', help='Run in test mode.')
parser.add_argument('--test_outer_T', type=int, help='Iteration during testing.')
parser.add_argument('--test_batch_size', type=int, help='Batch size during testing.')
parser.add_argument('--data_size', type=int, help='Batch size for training.')
parser.add_argument('--feas_rest', action='store_true', help='Restorate the feasibility.')
parser.add_argument('--feas_rest_num', type=int, help='The iteration number of feasibility restoration.')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay rate.')

# learning-based model
args, _ = parser.parse_known_args()
if args.model_name == 'LSTM':
    model = LSTM(args.num_eq + args.num_ineq, args.input_dim, args.hidden_dim, args.outer_T, args.device)
    params_save_dir = os.path.join(args.save_dir, model.name(), 'params')
    os.makedirs(params_save_dir, exist_ok=True)
    figs_save_dir = os.path.join(args.save_dir, model.name(), 'figs')
    os.makedirs(figs_save_dir, exist_ok=True)

# The method of Stage II
exact_model = LU(args.device)


# model parameter save path and data load path
if args.prob_type == 'QP_RHS':
    data_path = './datasets/QP_RHS_{}_{}_{}/'.format(args.num_var, args.num_ineq, args.num_eq)

    save_path = os.path.join(args.save_dir, model.name(), 'params', 'QP_RHS_{}_{}_{}_{}_{}.pth'.format(args.num_var,
                                                                                                             args.num_ineq,
                                                                                                             args.num_eq,
                                                                                                             args.outer_T,
                                                                                                             args.hidden_dim))
elif args.prob_type == 'QP':
    data_path = './datasets/QP_{}_{}_{}/'.format(args.num_var, args.num_ineq, args.num_eq)
    save_path = os.path.join(args.save_dir, model.name(), 'params', 'QP_{}_{}_{}_{}_{}.pth'.format(args.num_var,
                                                                                                       args.num_ineq,
                                                                                                       args.num_eq,
                                                                                                       args.outer_T,
                                                                                                       args.hidden_dim))
elif args.prob_type == 'Random_QP':
    data_path = './datasets/Random_QP_{}_{}/'.format(args.num_var, args.num_ineq)
    save_path = os.path.join(args.save_dir, model.name(), 'params', 'Random_QP_{}_{}_{}_{}.pth'.format(args.num_var,
                                                                                                       args.num_ineq,
                                                                                                       args.outer_T,
                                                                                                       args.hidden_dim))
elif args.prob_type == 'Equality_QP':
    data_path = './datasets/Equality_QP_{}_{}/'.format(args.num_var, args.num_eq)
    save_path = os.path.join(args.save_dir, model.name(), 'params', 'Equality_QP_{}_{}_{}_{}.pth'.format(args.num_var,
                                                                                                       args.num_eq,
                                                                                                       args.outer_T,
                                                                                                       args.hidden_dim))
elif args.prob_type == 'SVM':
    data_path = './datasets/SVM_{}_{}/'.format(args.num_var, args.num_ineq)
    save_path = os.path.join(args.save_dir, model.name(), 'params', 'SVM_{}_{}_{}_{}.pth'.format(args.num_var,
                                                                                                       args.num_ineq,
                                                                                                       args.outer_T,
                                                                                                       args.hidden_dim))
elif args.prob_type == 'QPLIB':
    data_path = './datasets/{}_{}'.format(args.prob_type, args.qplib_num)
    print('The problem being solved is {}: {}.'.format(args.prob_type, args.qplib_num))
    save_path = os.path.join(args.save_dir, model.name(), 'params', '{}_{}_{}_{}.pth'.format(args.prob_type,
                                                                                            args.qplib_num,
                                                                                            args.outer_T,
                                                                                            args.hidden_dim))
elif args.prob_type == 'MM_MOSARQP2':
    data_path = './datasets/{}'.format(args.prob_type)
    print('The problem being solved is {}.'.format(args.prob_type))
    save_path = os.path.join(args.save_dir, model.name(), 'params', '{}_{}_{}.pth'.format(args.prob_type,
                                                                                            args.outer_T,
                                                                                            args.hidden_dim))
elif args.prob_type == 'MM_QSCRS8':
    data_path = './datasets/{}'.format(args.prob_type)
    print('The problem being solved is {}.'.format(args.prob_type))
    save_path = os.path.join(args.save_dir, model.name(), 'params', '{}_{}_{}.pth'.format(args.prob_type,
                                                                                            args.outer_T,
                                                                                            args.hidden_dim))
elif args.prob_type == 'MM_QSCSD6':
    data_path = './datasets/{}'.format(args.prob_type)
    print('The problem being solved is {}.'.format(args.prob_type))
    save_path = os.path.join(args.save_dir, model.name(), 'params', '{}_{}_{}.pth'.format(args.prob_type,
                                                                                            args.outer_T,
                                                                                            args.hidden_dim))
elif args.prob_type == 'MM_Q25FV47':
    data_path = './datasets/{}'.format(args.prob_type)
    print('The problem being solved is {}.'.format(args.prob_type))
    save_path = os.path.join(args.save_dir, model.name(), 'params', '{}_{}_{}.pth'.format(args.prob_type,
                                                                                            args.outer_T,
                                                                                            args.hidden_dim))
elif args.prob_type == 'MM_QSHIP04L':
    data_path = './datasets/{}'.format(args.prob_type)
    print('The problem being solved is {}.'.format(args.prob_type))
    save_path = os.path.join(args.save_dir, model.name(), 'params', '{}_{}_{}.pth'.format(args.prob_type,
                                                                                            args.outer_T,
                                                                                            args.hidden_dim))

elif args.prob_type == 'MM_QSHIP08S':
    data_path = './datasets/{}'.format(args.prob_type)
    print('The problem being solved is {}.'.format(args.prob_type))
    save_path = os.path.join(args.save_dir, model.name(), 'params', '{}_{}_{}.pth'.format(args.prob_type,
                                                                                            args.outer_T,
                                                                                            args.hidden_dim))
elif args.prob_type == 'MM_CVXQP1_M':
    data_path = './datasets/{}'.format(args.prob_type)
    print('The problem being solved is {}.'.format(args.prob_type))
    save_path = os.path.join(args.save_dir, model.name(), 'params', '{}_{}_{}.pth'.format(args.prob_type,
                                                                                            args.outer_T,
                                                                                            args.hidden_dim))
elif args.prob_type == 'MM_CVXQP3_M':
    data_path = './datasets/{}'.format(args.prob_type)
    print('The problem being solved is {}.'.format(args.prob_type))
    save_path = os.path.join(args.save_dir, model.name(), 'params', '{}_{}_{}.pth'.format(args.prob_type,
                                                                                            args.outer_T,
                                                                                            args.hidden_dim))




random.seed(args.seed)

train_frac = 1 - args.val_frac - args.test_frac
train_size = int(args.data_size * train_frac)
val_size = int(args.data_size * args.val_frac)
test_size = args.data_size - train_size - val_size

#shuffle the number of datasets
dataset_ids = list(range(args.data_size))
random.shuffle(dataset_ids)
train_ids = dataset_ids[:train_size]
val_ids = dataset_ids[train_size:train_size+val_size]
test_ids = dataset_ids[train_size+val_size:]



if not args.test:
    stopper = EarlyStopping(save_path, patience=args.patience)  # Early stop detector

    # meta optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # training
    for epoch in range(args.num_epoch):
        model.train()
        train_start_time = time.time()

        for batch_i in range(int(train_size/args.batch_size)):
            # load data
            train_batch_ids = train_ids[batch_i*args.batch_size:(batch_i+1)*args.batch_size]
            Q, p, G, c, A, b, lb, ub, A0, zl, zu, x_gt, y_gt = [], [], [], [], [], [], [], [], [], [], [], [], []
            for j in range(len(train_batch_ids)):
                if args.prob_type == 'QP_RHS':
                    gz_name = 'qp_rhs_{}.gz'.format(train_batch_ids[j])
                elif args.prob_type == 'QP':
                    gz_name = 'qp_{}.gz'.format(train_batch_ids[j])
                elif args.prob_type == 'Random_QP':
                    gz_name = 'random_qp_{}.gz'.format(train_batch_ids[j])
                elif args.prob_type == 'Equality_QP':
                    gz_name = 'equality_qp_{}.gz'.format(train_batch_ids[j])
                elif args.prob_type == 'SVM':
                    gz_name = 'svm_{}.gz'.format(train_batch_ids[j])
                elif args.prob_type == 'QPLIB':
                    gz_name = 'qplib_{}_{}.gz'.format(args.qplib_num, train_batch_ids[j])
                elif args.prob_type == 'MM_MOSARQP2':
                    gz_name = 'mosarqp2_{}.gz'.format(train_batch_ids[j])
                elif args.prob_type == 'MM_QSCSD6':
                    gz_name = 'qscsd6_{}.gz'.format(train_batch_ids[j])
                elif args.prob_type == 'MM_QSCRS8':
                    gz_name = 'qscrs8_{}.gz'.format(train_batch_ids[j])
                elif args.prob_type == 'MM_Q25FV47':
                    gz_name = 'q25fv47_{}.gz'.format(train_batch_ids[j])
                elif args.prob_type == 'MM_QSHIP04L':
                    gz_name = 'qship04l_{}.gz'.format(train_batch_ids[j])
                elif args.prob_type == 'MM_QSHIP08S':
                    gz_name = 'qship08s_{}.gz'.format(train_batch_ids[j])
                elif args.prob_type == 'MM_CVXQP1_M':
                    gz_name = 'cvxqp1_m_{}.gz'.format(train_batch_ids[j])
                elif args.prob_type == 'MM_CVXQP3_M':
                    gz_name = 'cvxqp3_m_{}.gz'.format(train_batch_ids[j])


                gz_path = os.path.join(data_path, gz_name)

                with gzip.open(gz_path, 'rb') as f:
                    gz_dict = pickle.load(f)

                if (args.prob_type == 'QP') or (args.prob_type == 'QP_RHS'):
                    Q.append(gz_dict['Q'])
                    p.append(gz_dict['p'])
                    num_var = gz_dict['Q'].shape[1]
                else:
                    Q.append(gz_dict['Q'].toarray())
                    p.append(gz_dict['p'].toarray())
                    num_var = gz_dict['Q'].toarray().shape[1]


                try:
                    if (args.prob_type == 'QP') or (args.prob_type == 'QP_RHS'):
                        G.append(gz_dict['G'])
                        c.append(gz_dict['c'])
                        num_ineq = gz_dict['G'].shape[0]
                    else:
                        G.append(gz_dict['G'].toarray())
                        c.append(gz_dict['c'].toarray())
                        num_ineq = gz_dict['G'].toarray().shape[0]

                except KeyError:
                    num_ineq = 0

                try:
                    if (args.prob_type == 'QP') or (args.prob_type == 'QP_RHS'):
                        A.append(gz_dict['A'])
                        b.append(gz_dict['b'])
                        num_eq = gz_dict['A'].shape[0]
                    else:
                        A.append(gz_dict['A'].toarray())
                        b.append(gz_dict['b'].toarray())
                        num_eq = gz_dict['A'].toarray().shape[0]

                except KeyError:
                    num_eq = 0

                try:
                    if (args.prob_type == 'QP') or (args.prob_type == 'QP_RHS'):
                        lb.append(gz_dict['lb'])
                        ub.append(gz_dict['ub'])
                        num_lb = gz_dict['lb'].shape[0]
                        num_ub = gz_dict['ub'].shape[0]
                    else:
                        lb.append(gz_dict['lb'].toarray())
                        ub.append(gz_dict['ub'].toarray())
                        num_lb = gz_dict['lb'].toarray().shape[0]
                        num_ub = gz_dict['ub'].toarray().shape[0]
                except KeyError:
                    num_lb = 0
                    num_ub = 0

                if (args.prob_type == 'QP') or (args.prob_type == 'QP_RHS'):
                    A0.append(gz_dict['A0'])
                    zl.append(gz_dict['zl'])
                    zu.append(gz_dict['zu'])
                else:
                    A0.append(gz_dict['A0'].toarray())
                    zl.append(gz_dict['zl'].toarray())
                    zu.append(gz_dict['zu'].toarray())

            train_Q = torch.tensor(np.array(Q), dtype=torch.float32, device=args.device)*2
            train_p = torch.tensor(np.array(p), dtype=torch.float32, device=args.device)
            train_A0 = torch.tensor(np.array(A0), dtype=torch.float32, device=args.device)
            train_zl = torch.tensor(np.array(zl), dtype=torch.float32, device=args.device)
            train_zu = torch.tensor(np.array(zu), dtype=torch.float32, device=args.device)

            num_constr = train_A0.shape[1]

            if args.scaling:
                scaling = Scaling(num_var, num_constr, args.scaling_ites, args.device)
                train_Q_pre = train_Q
                train_p_pre = train_p
                train_Q, train_p, train_A0, train_zl, train_zu = scaling.scale_data(train_Q, train_p, train_A0, train_zl, train_zu)

            if num_ineq != 0:
                train_G = torch.tensor(np.array(G), dtype=torch.float32, device=args.device)
                train_c = torch.tensor(np.array(c), dtype=torch.float32, device=args.device)
            if num_eq != 0:
                train_A = torch.tensor(np.array(A), dtype=torch.float32, device=args.device)
                train_b = torch.tensor(np.array(b), dtype=torch.float32, device=args.device)
            if num_lb != 0:
                train_lb = torch.tensor(np.array(lb), dtype=torch.float32, device=args.device)
            else:
                train_lb = None
            if num_ub != 0:
                train_ub = torch.tensor(np.array(ub), dtype=torch.float32, device=args.device)
            else:
                train_ub = None

            batch_size = train_p.shape[0]
            train_x = torch.zeros((batch_size, num_var, 1), device=args.device)
            train_y = torch.zeros((batch_size, num_constr, 1), device=args.device)
            train_z = torch.zeros((batch_size, num_constr, 1), device=args.device)
            train_xv = torch.zeros((batch_size, num_var + num_constr, 1), device=args.device)
            train_H = torch.zeros((batch_size, num_var + num_constr, args.hidden_dim), device=args.device)
            train_C = torch.zeros((batch_size, num_var + num_constr, args.hidden_dim), device=args.device)
            train_sigma = args.sigma

            for truc_i in range(int(args.outer_T/args.truncated_length)):
                train_loss = 0.0
                for t in range(args.truncated_length):
                    train_x, train_y, train_z, train_xv, train_H, train_C, _, _, _ = model(t, num_ineq, num_eq,
                                                                                        train_x, train_y, train_z,
                                                                                        train_xv, train_sigma, train_H, train_C,
                                                                                        Q=train_Q, p=train_p, A0=train_A0,
                                                                                        lb=train_lb, ub=train_ub,
                                                                                        zl=train_zl, zu=train_zu)

                    train_primal_res, train_dual_res, loss = primal_dual_loss(train_x, train_y, train_z, train_Q, train_p, train_A0)
                    train_loss += loss.mean()/args.outer_T

                optimizer.zero_grad()
                train_loss.backward(retain_graph=True)
                optimizer.step()

                train_x = train_x.detach()
                train_y = train_y.detach()
                train_z = train_z.detach()
                train_xv = train_xv.detach()
                train_H = train_H.detach()
                train_C = train_C.detach()

        train_end_time = time.time()

        if args.scaling:
            train_Q = train_Q_pre
            train_p = train_p_pre
            train_x = torch.bmm(scaling.D, train_x)

        train_obj = obj_fn(train_x, Q=train_Q, p=train_p).mean()
        if num_ineq != 0:
            train_ineq_vio_max = ineq_dist(train_x, G=train_G, c=train_c).max(dim=1).values.mean()
            train_ineq_vio_mean = ineq_dist(train_x, G=train_G, c=train_c).mean()
        if num_eq != 0:
            train_eq_vio_max = eq_dist(train_x, A=train_A, b=train_b).max(dim=1).values.mean()
            train_eq_vio_mean = eq_dist(train_x, A=train_A, b=train_b).mean()
        if num_lb != 0:
            train_lb_vio_max = lb_dist(train_x, lb=train_lb).max(dim=1).values.mean()
            train_lb_vio_mean = lb_dist(train_x, lb=train_lb).mean()
        if num_ub != 0:
            train_ub_vio_max = ub_dist(train_x, ub=train_ub).max(dim=1).values.mean()
            train_ub_vio_mean = ub_dist(train_x, ub=train_ub).mean()

        # validation
        model.eval()
        with torch.no_grad():
            val_Q, val_p, val_G, val_c, val_A, val_b, val_lb, val_ub, val_A0, val_zl, val_zu = [], [], [], [], [], [], [], [], [], [], []
            for k in range(len(val_ids)):
                if args.prob_type == 'QP_RHS':
                    gz_name = 'qp_rhs_{}.gz'.format(val_ids[k])
                elif args.prob_type == 'QP':
                    gz_name = 'qp_{}.gz'.format(val_ids[k])
                elif args.prob_type == 'Random_QP':
                    gz_name = 'random_qp_{}.gz'.format(val_ids[k])
                elif args.prob_type == 'Equality_QP':
                    gz_name = 'equality_qp_{}.gz'.format(val_ids[k])
                elif args.prob_type == 'SVM':
                    gz_name = 'svm_{}.gz'.format(val_ids[k])
                elif args.prob_type == 'QPLIB':
                    gz_name = 'qplib_{}_{}.gz'.format(args.qplib_num, val_ids[k])
                elif args.prob_type == 'MM_MOSARQP2':
                    gz_name = 'mosarqp2_{}.gz'.format(val_ids[k])
                elif args.prob_type == 'MM_QSCSD6':
                    gz_name = 'qscsd6_{}.gz'.format(val_ids[k])
                elif args.prob_type == 'MM_QSCRS8':
                    gz_name = 'qscrs8_{}.gz'.format(val_ids[k])
                elif args.prob_type == 'MM_Q25FV47':
                    gz_name = 'q25fv47_{}.gz'.format(val_ids[k])
                elif args.prob_type == 'MM_QSHIP04L':
                    gz_name = 'qship04l_{}.gz'.format(val_ids[k])
                elif args.prob_type == 'MM_QSHIP08S':
                    gz_name = 'qship08s_{}.gz'.format(val_ids[k])
                elif args.prob_type == 'MM_CVXQP1_M':
                    gz_name = 'cvxqp1_m_{}.gz'.format(val_ids[k])
                elif args.prob_type == 'MM_CVXQP3_M':
                    gz_name = 'cvxqp3_m_{}.gz'.format(val_ids[k])


                gz_path = os.path.join(data_path, gz_name)
                with gzip.open(gz_path, 'rb') as f:
                    gz_dict = pickle.load(f)

                if (args.prob_type == 'QP') or (args.prob_type == 'QP_RHS'):
                    val_Q.append(gz_dict['Q'])
                    val_p.append(gz_dict['p'])
                else:
                    val_Q.append(gz_dict['Q'].toarray())
                    val_p.append(gz_dict['p'].toarray())

                try:
                    if (args.prob_type == 'QP') or (args.prob_type == 'QP_RHS'):
                        val_G.append(gz_dict['G'])
                        val_c.append(gz_dict['c'])
                    else:
                        val_G.append(gz_dict['G'].toarray())
                        val_c.append(gz_dict['c'].toarray())
                except KeyError:
                    num_ineq = 0

                try:
                    if (args.prob_type == 'QP') or (args.prob_type == 'QP_RHS'):
                        val_A.append(gz_dict['A'])
                        val_b.append(gz_dict['b'])
                    else:
                        val_A.append(gz_dict['A'].toarray())
                        val_b.append(gz_dict['b'].toarray())
                except KeyError:
                    num_eq = 0

                try:
                    if (args.prob_type == 'QP') or (args.prob_type == 'QP_RHS'):
                        val_lb.append(gz_dict['lb'])
                        val_ub.append(gz_dict['ub'])
                    else:
                        val_lb.append(gz_dict['lb'].toarray())
                        val_ub.append(gz_dict['ub'].toarray())
                except KeyError:
                    num_lb = 0
                    num_ub = 0

                if (args.prob_type == 'QP') or (args.prob_type == 'QP_RHS'):
                    val_A0.append(gz_dict['A0'])
                    val_zl.append(gz_dict['zl'])
                    val_zu.append(gz_dict['zu'])
                else:
                    val_A0.append(gz_dict['A0'].toarray())
                    val_zl.append(gz_dict['zl'].toarray())
                    val_zu.append(gz_dict['zu'].toarray())

            val_Q = torch.tensor(np.array(val_Q), dtype=torch.float32, device=args.device)*2
            val_p = torch.tensor(np.array(val_p), dtype=torch.float32, device=args.device)
            val_A0 = torch.tensor(np.array(val_A0), dtype=torch.float32, device=args.device)
            val_zl = torch.tensor(np.array(val_zl), dtype=torch.float32, device=args.device)
            val_zu = torch.tensor(np.array(val_zu), dtype=torch.float32, device=args.device)

            if args.scaling:
                scaling = Scaling(num_var, num_constr, args.scaling_ites, args.device)
                val_Q_pre = val_Q
                val_p_pre = val_p
                val_Q, val_p, val_A0, val_zl, val_zu = scaling.scale_data(val_Q, val_p, val_A0, val_zl, val_zu)

            if num_ineq != 0:
                val_G = torch.tensor(np.array(val_G), dtype=torch.float32, device=args.device)
                val_c = torch.tensor(np.array(val_c), dtype=torch.float32, device=args.device)
            if num_eq != 0:
                val_A = torch.tensor(np.array(val_A), dtype=torch.float32, device=args.device)
                val_b = torch.tensor(np.array(val_b), dtype=torch.float32, device=args.device)
            if num_lb != 0:
                val_lb = torch.tensor(np.array(val_lb), dtype=torch.float32, device=args.device)
            else:
                val_lb = None
            if num_ub != 0:
                val_ub = torch.tensor(np.array(val_ub), dtype=torch.float32, device=args.device)
            else:
                val_ub = None

            val_x = torch.zeros((val_size, num_var, 1), device=args.device)
            val_y = torch.zeros((val_size, num_constr, 1), device=args.device)
            val_z = torch.zeros((val_size, num_constr, 1), device=args.device)
            val_xv = torch.zeros((val_size, num_var + num_constr, 1), device=args.device)
            val_H = torch.zeros((val_size, num_var + num_constr, args.hidden_dim), device=args.device)
            val_C = torch.zeros((val_size, num_var + num_constr, args.hidden_dim), device=args.device)
            val_sigma = args.sigma

            val_start_time = time.time()
            for t in range(args.outer_T):
                val_x, val_y, val_z, val_xv, val_H, val_C, _, _, _ = model(t, num_ineq, num_eq,
                                                                        val_x, val_y, val_z,
                                                                        val_xv, val_sigma, val_H, val_C,
                                                                        Q=val_Q, p=val_p, A0=val_A0,
                                                                        lb=val_lb, ub=val_ub,
                                                                        zl=val_zl, zu=val_zu)
            val_end_time = time.time()

            if args.scaling:
                val_Q = val_Q_pre
                val_p = val_p_pre
                val_x = torch.bmm(scaling.D, val_x)

            val_vios = []
            val_obj = obj_fn(val_x, Q=val_Q, p=val_p).mean()
            if num_ineq != 0:
                val_ineq_vio_max = ineq_dist(val_x, G=val_G, c=val_c).max(dim=1).values.mean()
                val_ineq_vio_mean = ineq_dist(val_x, G=val_G, c=val_c).mean()
                val_vios.append(val_ineq_vio_max.data.item())
            if num_eq != 0:
                val_eq_vio_max = eq_dist(val_x, A=val_A, b=val_b).max(dim=1).values.mean()
                val_eq_vio_mean = eq_dist(val_x, A=val_A, b=val_b).mean()
                val_vios.append(val_eq_vio_max.data.item())
            if num_lb != 0:
                val_lb_vio_max = lb_dist(val_x, lb=val_lb).max(dim=1).values.mean()
                val_lb_vio_mean = lb_dist(val_x, lb=val_lb).mean()
                val_vios.append(val_lb_vio_max.data.item())
            if num_ub != 0:
                val_ub_vio_max = ub_dist(val_x, ub=val_ub).max(dim=1).values.mean()
                val_ub_vio_mean = ub_dist(val_x, ub=val_ub).mean()
                val_vios.append(val_ub_vio_max.data.item())

        early_stop = stopper.step(val_obj.data.item(), model, args.early_stop_mode, args.eq_tol, *val_vios)
        print("Epoch : {} | Train_Obj : {:.3f} | Val_Obj : {:.3f} | Train_Time : {:.3f} | Val_Time : {:.3f} |".format(epoch, train_obj, val_obj, train_end_time - train_start_time, val_end_time - val_start_time))
        if num_ineq != 0:
            print("Epoch : {} | Train_Max_Ineq : {:.3f} | Train_Mean_Ineq : {:.3f} | Val_Max_Ineq : {:.3f} | Val_Mean_Ineq : {:.3f} |".format(epoch, train_ineq_vio_max, train_ineq_vio_mean, val_ineq_vio_max, val_ineq_vio_mean))
        if num_eq != 0:
            print("Epoch : {} | Train_Max_Eq : {:.3f} | Train_Mean_Eq : {:.3f} | Val_Max_Eq : {:.3f} | Val_Mean_Eq : {:.3f} |".format(epoch, train_eq_vio_max, train_eq_vio_mean, val_eq_vio_max, val_eq_vio_mean))
        if num_lb != 0:
            print("Epoch : {} | Train_Max_Lb : {:.3f} | Train_Mean_Lb : {:.3f} | Val_Max_Lb : {:.3f} | Val_Mean_Lb : {:.3f} |".format(epoch, train_lb_vio_max, train_lb_vio_mean, val_lb_vio_max, val_lb_vio_mean))
        if num_ub != 0:
            print("Epoch : {} | Train_Max_Ub : {:.3f} | Train_Mean_Ub : {:.3f} | Val_Max_Ub : {:.3f} | Val_Mean_Ub : {:.3f} |".format(epoch, train_ub_vio_max, train_ub_vio_mean, val_ub_vio_max, val_ub_vio_mean))
        if early_stop:
            break

elif args.test:
    if args.prob_type == 'QP_RHS':
        load_path = os.path.join(args.save_dir, model.name(), 'params', 'QP_RHS_{}_{}_{}_{}_{}.pth'.format(args.num_var,
                                                                                                           args.num_eq,
                                                                                                           args.num_ineq,
                                                                                                           args.outer_T,
                                                                                                           args.hidden_dim))
    elif args.prob_type == 'QP':
        load_path = os.path.join(args.save_dir, model.name(), 'params', 'QP_{}_{}_{}_{}_{}.pth'.format(args.num_var,
                                                                                                           args.num_eq,
                                                                                                           args.num_ineq,
                                                                                                           args.outer_T,
                                                                                                           args.hidden_dim))
    elif args.prob_type == 'Random_QP':
        load_path = os.path.join(args.save_dir, model.name(), 'params', 'Random_QP_{}_{}_{}_{}.pth'.format(args.num_var,
                                                                                                           args.num_ineq,
                                                                                                           args.outer_T,
                                                                                                           args.hidden_dim))
    elif args.prob_type == 'Equality_QP':
        data_path = './datasets/Equality_QP_{}_{}/'.format(args.num_var, args.num_eq)
        load_path = os.path.join(args.save_dir, model.name(), 'params', 'Equality_QP_{}_{}_{}_{}.pth'.format(args.num_var,
                                                                                                                args.num_eq,
                                                                                                                args.outer_T,
                                                                                                                args.hidden_dim))
    elif args.prob_type == 'QPLIB':
        load_path = os.path.join(args.save_dir, model.name(), 'params', '{}_{}_{}_{}.pth'.format(args.prob_type,
                                                                                                 args.qplib_num,
                                                                                                 args.outer_T,
                                                                                                 args.hidden_dim))
    elif args.prob_type == 'SVM':
        data_path = './datasets/SVM_{}_{}/'.format(args.num_var, args.num_ineq)
        load_path = os.path.join(args.save_dir, model.name(), 'params', 'SVM_{}_{}_{}_{}.pth'.format(args.num_var,
                                                                                                            args.num_ineq,
                                                                                                            args.outer_T,
                                                                                                            args.hidden_dim))
    elif args.prob_type == 'MM_MOSARQP2':
        load_path = os.path.join(args.save_dir, model.name(), 'params', '{}_{}_{}.pth'.format(args.prob_type,
                                                                                                 args.outer_T,
                                                                                                 args.hidden_dim))
    elif args.prob_type == 'MM_QSCSD6':
        load_path = os.path.join(args.save_dir, model.name(), 'params', '{}_{}_{}.pth'.format(args.prob_type,
                                                                                                 args.outer_T,
                                                                                                 args.hidden_dim))
    elif args.prob_type == 'MM_QSCRS8':
        load_path = os.path.join(args.save_dir, model.name(), 'params', '{}_{}_{}.pth'.format(args.prob_type,
                                                                                                 args.outer_T,
                                                                                                 args.hidden_dim))
    elif args.prob_type == 'MM_Q25FV47':
        load_path = os.path.join(args.save_dir, model.name(), 'params', '{}_{}_{}.pth'.format(args.prob_type,
                                                                                                 args.outer_T,
                                                                                                 args.hidden_dim))
    elif args.prob_type == 'MM_QSHIP04L':
        load_path = os.path.join(args.save_dir, model.name(), 'params', '{}_{}_{}.pth'.format(args.prob_type,
                                                                                                 args.outer_T,
                                                                                                 args.hidden_dim))
    elif args.prob_type == 'MM_QSHIP08S':
        load_path = os.path.join(args.save_dir, model.name(), 'params', '{}_{}_{}.pth'.format(args.prob_type,
                                                                                                 args.outer_T,
                                                                                                 args.hidden_dim))
    elif args.prob_type == 'MM_CVXQP1_M':
        load_path = os.path.join(args.save_dir, model.name(), 'params', '{}_{}_{}.pth'.format(args.prob_type,
                                                                                                 args.outer_T,
                                                                                                 args.hidden_dim))
    elif args.prob_type == 'MM_CVXQP3_M':
        load_path = os.path.join(args.save_dir, model.name(), 'params', '{}_{}_{}.pth'.format(args.prob_type,
                                                                                                 args.outer_T,
                                                                                                 args.hidden_dim))


    model.load_state_dict(torch.load(load_path))
    model.eval()
    with torch.no_grad():
        test_Q_all, test_p_all, test_G_all, test_c_all, test_A_all, test_b_all, test_lb_all, test_ub_all, test_A0_all, test_zl_all, test_zu_all = [], [], [], [], [], [], [], [], [], [], []
        for k in range(len(test_ids)):
            if args.prob_type == 'QP_RHS':
                gz_name = 'qp_rhs_{}.gz'.format(test_ids[k])
            elif args.prob_type == 'QP':
                gz_name = 'qp_{}.gz'.format(test_ids[k])
            elif args.prob_type == 'Random_QP':
                gz_name = 'random_qp_{}.gz'.format(test_ids[k])
            elif args.prob_type == 'Equality_QP':
                gz_name = 'equality_qp_{}.gz'.format(test_ids[k])
            elif args.prob_type == 'SVM':
                gz_name = 'svm_{}.gz'.format(test_ids[k])
            elif args.prob_type == 'QPLIB':
                gz_name = 'qplib_{}_{}.gz'.format(args.qplib_num, test_ids[k])
            elif args.prob_type == 'MM_MOSARQP2':
                gz_name = 'mosarqp2_{}.gz'.format(test_ids[k])
            elif args.prob_type == 'MM_QSCSD6':
                gz_name = 'qscsd6_{}.gz'.format(test_ids[k])
            elif args.prob_type == 'MM_QSCRS8':
                gz_name = 'qscrs8_{}.gz'.format(test_ids[k])
            elif args.prob_type == 'MM_Q25FV47':
                gz_name = 'q25fv47_{}.gz'.format(test_ids[k])
            elif args.prob_type == 'MM_QSHIP04L':
                gz_name = 'qship04l_{}.gz'.format(test_ids[k])
            elif args.prob_type == 'MM_QSHIP08S':
                gz_name = 'qship08s_{}.gz'.format(test_ids[k])
            elif args.prob_type == 'MM_CVXQP1_M':
                gz_name = 'cvxqp1_m_{}.gz'.format(test_ids[k])
            elif args.prob_type == 'MM_CVXQP3_M':
                gz_name = 'cvxqp3_m_{}.gz'.format(test_ids[k])


            gz_path = os.path.join(data_path, gz_name)
            with gzip.open(gz_path, 'rb') as f:
                gz_dict = pickle.load(f)

            if (args.prob_type == 'QP') or (args.prob_type == 'QP_RHS'):
                test_Q_all.append(gz_dict['Q'])
                test_p_all.append(gz_dict['p'])
                num_var = gz_dict['Q'].shape[1]
            else:
                test_Q_all.append(gz_dict['Q'].toarray())
                test_p_all.append(gz_dict['p'].toarray())
                num_var = gz_dict['Q'].toarray().shape[1]

            try:
                if (args.prob_type == 'QP') or (args.prob_type == 'QP_RHS'):
                    test_G_all.append(gz_dict['G'])
                    test_c_all.append(gz_dict['c'])
                    num_ineq = gz_dict['G'].shape[0]
                else:
                    test_G_all.append(gz_dict['G'].toarray())
                    test_c_all.append(gz_dict['c'].toarray())
                    num_ineq = gz_dict['G'].toarray().shape[0]

            except KeyError:
                num_ineq = 0

            try:
                if (args.prob_type == 'QP') or (args.prob_type == 'QP_RHS'):
                    test_A_all.append(gz_dict['A'])
                    test_b_all.append(gz_dict['b'])
                    num_eq = gz_dict['A'].shape[0]
                else:
                    test_A_all.append(gz_dict['A'].toarray())
                    test_b_all.append(gz_dict['b'].toarray())
                    num_eq = gz_dict['A'].shape[0]

            except KeyError:
                num_eq = 0

            try:
                if (args.prob_type == 'QP') or (args.prob_type == 'QP_RHS'):
                    test_lb_all.append(gz_dict['lb'])
                    test_ub_all.append(gz_dict['ub'])
                    num_lb = gz_dict['lb'].shape[0]
                    num_ub = gz_dict['ub'].shape[0]
                else:
                    test_lb_all.append(gz_dict['lb'].toarray())
                    test_ub_all.append(gz_dict['ub'].toarray())
                    num_lb = gz_dict['lb'].toarray().shape[0]
                    num_ub = gz_dict['ub'].toarray().shape[0]

            except KeyError:
                num_lb = 0
                num_ub = 0

            if (args.prob_type == 'QP') or (args.prob_type == 'QP_RHS'):
                test_A0_all.append(gz_dict['A0'])
                test_zl_all.append(gz_dict['zl'])
                test_zu_all.append(gz_dict['zu'])
            else:
                test_A0_all.append(gz_dict['A0'].toarray())
                test_zl_all.append(gz_dict['zl'].toarray())
                test_zu_all.append(gz_dict['zu'].toarray())


        test_Q_all = torch.tensor(np.array(test_Q_all), dtype=torch.float32, device=args.device) * 2
        test_p_all = torch.tensor(np.array(test_p_all), dtype=torch.float32, device=args.device)
        test_A0_all = torch.tensor(np.array(test_A0_all), dtype=torch.float32, device=args.device)
        test_zl_all = torch.tensor(np.array(test_zl_all), dtype=torch.float32, device=args.device)
        test_zu_all = torch.tensor(np.array(test_zu_all), dtype=torch.float32, device=args.device)

        num_constr = test_A0_all.shape[1]

        if num_ineq != 0:
            test_G_all = torch.tensor(np.array(test_G_all), dtype=torch.float32, device=args.device)
            test_c_all = torch.tensor(np.array(test_c_all), dtype=torch.float32, device=args.device)
        if num_eq != 0:
            test_A_all = torch.tensor(np.array(test_A_all), dtype=torch.float32, device=args.device)
            test_b_all = torch.tensor(np.array(test_b_all), dtype=torch.float32, device=args.device)
        if num_lb != 0:
            test_lb_all = torch.tensor(np.array(test_lb_all), dtype=torch.float32, device=args.device)
        else:
            test_lb_all = None
        if num_ub != 0:
            test_ub_all = torch.tensor(np.array(test_ub_all), dtype=torch.float32, device=args.device)
        else:
            test_ub_all = None

        test_objs = []
        test_objs_fr = []
        if num_ineq != 0:
            test_ineq_vio_maxs = []
            test_ineq_vio_means = []
            test_ineq_vio_maxs_fr = []
            test_ineq_vio_means_fr = []
        if num_eq != 0:
            test_eq_vio_maxs = []
            test_eq_vio_means = []
            test_eq_vio_maxs_fr = []
            test_eq_vio_means_fr = []
        if num_lb != 0:
            test_lb_vio_maxs = []
            test_lb_vio_means = []
            test_lb_vio_maxs_fr = []
            test_lb_vio_means_fr = []
        if num_ub != 0:
            test_ub_vio_maxs = []
            test_ub_vio_means = []
            test_ub_vio_maxs_fr = []
            test_ub_vio_means_fr = []
        test_ls_residuals = []
        test_primal_residuals = []
        test_dual_residuals = []
        test_primal_residuals_fr = []
        test_dual_residuals_fr = []

        # theoretical conditions
        test_x_cond_1_left = []
        test_x_cond_1_right = []
        test_x_cond_2_left = []
        test_x_cond_2_right = []
        test_z_cond_1_left = []
        test_z_cond_1_right = []
        test_z_cond_2_left = []
        test_z_cond_2_right = []
        test_alpha_cond_left = []
        test_alpha_cond_right = []

        test_total_time = 0.0
        test_para_time = 0.0

        for i in range(int(test_size / args.test_batch_size)):

            test_Q = test_Q_all[i * args.test_batch_size:(i + 1) * args.test_batch_size,:,:]
            test_p = test_p_all[i * args.test_batch_size:(i + 1) * args.test_batch_size,:,:]

            if num_ineq != 0:
                test_G = test_G_all[i * args.test_batch_size:(i + 1) * args.test_batch_size, :, :]
                test_c = test_c_all[i * args.test_batch_size:(i + 1) * args.test_batch_size, :, :]
            else:
                test_G = None
                test_c = None

            if num_eq != 0:
                test_A = test_A_all[i * args.test_batch_size:(i + 1) * args.test_batch_size, :, :]
                test_b = test_b_all[i * args.test_batch_size:(i + 1) * args.test_batch_size, :, :]
            else:
                test_A = None
                test_b = None

            if num_lb != 0:
                test_lb = test_lb_all[i * args.test_batch_size:(i + 1) * args.test_batch_size, :, :]
            else:
                test_lb = None

            if num_ub != 0:
                test_ub = test_ub_all[i * args.test_batch_size:(i + 1) * args.test_batch_size, :, :]
            else:
                test_ub = None

            test_A0 = test_A0_all[i * args.test_batch_size:(i + 1) * args.test_batch_size, :, :]
            test_zl = test_zl_all[i * args.test_batch_size:(i + 1) * args.test_batch_size, :, :]
            test_zu = test_zu_all[i * args.test_batch_size:(i + 1) * args.test_batch_size, :, :]


            if args.scaling:
                scaling_time = 0.0
                test_Q_pre = test_Q
                test_p_pre = test_p
                test_A0_pre = test_A0
                test_zl_pre = test_zl
                test_zu_pre = test_zu
                start_time = time.time()
                scaling = Scaling(num_var, num_constr, args.scaling_ites, args.device)
                test_Q_scaling, test_p_scaling, test_A0_scaling, test_zl_scaling, test_zu_scaling = scaling.scale_data(test_Q, test_p, test_A0, test_zl, test_zu)
                test_Q = test_Q_scaling
                test_p = test_p_scaling
                test_A0 = test_A0_scaling
                test_zl = test_zl_scaling
                test_zu = test_zu_scaling
                end_time = time.time()
                scaling_time += (end_time-start_time)


            test_x = torch.zeros((args.test_batch_size, num_var, 1), device=args.device)
            test_y = torch.zeros((args.test_batch_size, num_constr, 1), device=args.device)
            test_z = torch.zeros((args.test_batch_size, num_constr, 1), device=args.device)
            test_xv = torch.zeros((args.test_batch_size, num_var + num_constr, 1), device=args.device)
            test_H = torch.zeros((args.test_batch_size, num_var + num_constr, args.hidden_dim), device=args.device)
            test_C = torch.zeros((args.test_batch_size, num_var + num_constr, args.hidden_dim), device=args.device)
            test_sigma = args.sigma

            objs = []
            if num_ineq != 0:
                ineq_vio_maxs = []
                ineq_vio_means = []
            if num_eq != 0:
                eq_vio_maxs = []
                eq_vio_means = []
            if num_lb != 0:
                lb_vio_maxs = []
                lb_vio_means = []
            if num_ub != 0:
                ub_vio_maxs = []
                ub_vio_means = []
            ls_residuals = []
            primal_residuals = []
            dual_residuals = []

            # conditions
            x_cond_1_left = []
            x_cond_1_right = []
            x_cond_2_left = []
            x_cond_2_right = []
            z_cond_1_left = []
            z_cond_1_right = []
            z_cond_2_left = []
            z_cond_2_right = []
            alpha_cond_left = []
            alpha_cond_right = []

            for t in range(args.test_outer_T):
                if args.scaling:
                    test_x_pre = torch.bmm(scaling.D, test_x)
                    test_y_pre = torch.bmm(scaling.cinv * scaling.E, test_y)
                    test_z_pre = torch.bmm(scaling.Einv, test_z)


                start_time = time.time()
                test_x, test_y, test_z, test_xv, test_H, test_C, test_A_tild, test_b_tild, test_rho_vec = model(t, num_ineq, num_eq,
                                                                                                            test_x, test_y, test_z,
                                                                                                            test_xv, test_sigma, test_H, test_C,
                                                                                                            Q=test_Q, p=test_p, A0=test_A0,
                                                                                                            lb=test_lb, ub=test_ub,
                                                                                                            zl=test_zl, zu=test_zu)

                end_time = time.time()
                test_total_time += (end_time-start_time)

                if args.scaling:
                    test_Q = test_Q_pre
                    test_p = test_p_pre
                    test_A0 = test_A0_pre
                    test_zl = test_zl_pre
                    test_zu = test_zu_pre
                    test_x_scaling = test_x
                    test_y_scaling = test_y
                    test_z_scaling = test_z
                    # if t>0:
                    #     # x subproblem condition 1
                    #     cx = 1
                    #     test_x_tild = torch.bmm(scaling.D, test_xv[:, :num_var, :])
                    #     sigma_Q_max = torch.linalg.eigvalsh(test_Q[0])[-1]
                    #     sigma_AA_min = torch.linalg.eigvalsh(test_A0[0].T@test_A0[0])[0]
                    #     rho_norm = torch.linalg.vector_norm(test_rho_vec, dim=(1,2), keepdim=True).mean()
                    #     beta_x = (2*1.1/0.9)*(2*(sigma_Q_max/rho_norm+cx)**2+8*(cx**2))/sigma_AA_min
                    #     x_diff = (torch.linalg.vector_norm(test_x_tild-test_x_pre, dim=(1,2), keepdim=True).mean())**2
                    #     x_cond_1_left.append(((rho_norm*x_diff*beta_x)/2+aug_lagr(test_x_tild, test_z_pre, test_y_pre, test_Q, test_p, test_A0, test_rho_vec).mean()).detach().cpu().numpy())
                    #
                    #     x subproblem condition 2
                    #     x_cond_2_left.append((torch.linalg.vector_norm(torch.bmm(test_Q, test_x_tild)+test_p+torch.bmm(test_A0.permute(0,2,1), test_y_pre)+
                    #                          torch.bmm(test_A0.permute(0,2,1), torch.bmm(torch.diag_embed(test_rho_vec.squeeze(-1)), torch.bmm(test_A0, test_x_tild)-test_z_pre)), dim=(1, 2), keepdim=True)).detach().cpu().numpy())
                    #     x_cond_2_right.append((cx*rho_norm*torch.linalg.vector_norm(test_x_tild-test_x_pre, dim=(1,2), keepdim=True)).detach().cpu().numpy())

                        # # z subproblem condition 1
                        # z_cond_1_right.append(aug_lagr(test_x_tild, test_z_pre, test_y_pre, test_Q, test_p, test_A0, test_rho_vec).mean().detach().cpu().numpy())



                    test_x = torch.bmm(scaling.D, test_x)
                    test_z = torch.bmm(scaling.Einv, test_z)
                    # if t>0:
                    #     # z subproblem condition 1
                    #     beta_z = (32 * 1.1) / ((sigma_AA_min ** 2) * 0.9)
                    #     z_diff = (torch.linalg.vector_norm(test_z-test_z_pre, dim=(1,2), keepdim=True).mean())**2
                    #     z_cond_1_left.append(((rho_norm*z_diff*beta_z)/2+aug_lagr(test_x_tild, test_z, test_y_pre, test_Q, test_p, test_A0, test_rho_vec).mean()).detach().cpu().numpy())
                    #
                    #     # z subproblem condition 2
                    #     mask_right = (test_z == test_zu) & ((test_y_pre + test_rho_vec*(torch.bmm(test_A0, test_x_tild) - test_z))>0)
                    #     mask_left = (test_z == test_zl) & ((test_y_pre + test_rho_vec * (torch.bmm(test_A0, test_x_tild) - test_z)) < 0)
                    #     z_part_grad = -test_y_pre-test_rho_vec*(torch.bmm(test_A0, test_x_tild)-test_z)
                    #     z_part_grad[mask_right] = 0
                    #     z_part_grad[mask_left] = 0
                    #     z_cond_2_left.append(torch.linalg.vector_norm(z_part_grad, dim=(1,2), keepdim=True).mean().detach().cpu().numpy())
                    #     cz = 1
                    #     z_cond_2_right.append((cz*rho_norm*(torch.linalg.vector_norm(test_z-test_z_pre, dim=(1,2), keepdim=True)+torch.linalg.vector_norm(test_x_tild-test_x_pre, dim=(1,2), keepdim=True))).mean().detach().cpu().numpy())

                    test_y = torch.bmm(scaling.cinv*scaling.E, test_y)
                    # if t>0:
                    #     alpha_cond_left.append(aug_lagr(test_x, test_z, test_y, test_Q, test_p, test_A0, test_rho_vec).mean().detach().cpu().numpy())
                    #     x_diff = (torch.linalg.vector_norm(test_x - test_x_tild, dim=(1, 2), keepdim=True).mean()) ** 2
                    #     alpha_cond_right.append((aug_lagr(test_x_tild, test_z, test_y, test_Q, test_p, test_A0, test_rho_vec).mean()-0.9*rho_norm*x_diff).detach().cpu().numpy())



                #objective values
                test_obj = obj_fn(test_x, Q=test_Q, p=test_p).mean()
                objs.append(test_obj.detach().cpu().numpy())
                #linear system residues
                test_ls_res = torch.linalg.vector_norm(torch.bmm(test_A_tild, test_xv)-test_b_tild, dim=(1,2), keepdim=True).mean()
                ls_residuals.append(test_ls_res.detach().cpu().numpy())
                #primal and dual residues
                test_primal_res, test_dual_res, loss = primal_dual_loss(test_x, test_y, test_z, test_Q, test_p, test_A0)
                primal_residuals.append(test_primal_res.mean().detach().cpu().numpy())
                dual_residuals.append(test_dual_res.mean().detach().cpu().numpy())

                if num_ineq != 0:
                    test_ineq_vio_max = ineq_dist(test_x, G=test_G, c=test_c).max(dim=1).values.mean()
                    test_ineq_vio_mean = ineq_dist(test_x, G=test_G, c=test_c).mean()
                    ineq_vio_maxs.append(test_ineq_vio_max.detach().cpu().numpy())
                    ineq_vio_means.append(test_ineq_vio_mean.detach().cpu().numpy())
                if num_eq != 0:
                    test_eq_vio_max = eq_dist(test_x, A=test_A, b=test_b).max(dim=1).values.mean()
                    test_eq_vio_mean = eq_dist(test_x, A=test_A, b=test_b).mean()
                    eq_vio_maxs.append(test_eq_vio_max.detach().cpu().numpy())
                    eq_vio_means.append(test_eq_vio_mean.detach().cpu().numpy())
                if num_lb != 0:
                    test_lb_vio_max = lb_dist(test_x, lb=test_lb).max(dim=1).values.mean()
                    test_lb_vio_mean = lb_dist(test_x, lb=test_lb).mean()
                    lb_vio_maxs.append(test_lb_vio_max.detach().cpu().numpy())
                    lb_vio_means.append(test_lb_vio_mean.detach().cpu().numpy())
                if num_ub != 0:
                    test_ub_vio_max = ub_dist(test_x, ub=test_ub).max(dim=1).values.mean()
                    test_ub_vio_mean = ub_dist(test_x, ub=test_ub).mean()
                    ub_vio_maxs.append(test_ub_vio_max.detach().cpu().numpy())
                    ub_vio_means.append(test_ub_vio_mean.detach().cpu().numpy())

                if args.scaling:
                    test_Q = test_Q_scaling
                    test_p = test_p_scaling
                    test_A0 = test_A0_scaling
                    test_zl = test_zl_scaling
                    test_zu = test_zu_scaling
                    test_x = test_x_scaling
                    test_y = test_y_scaling
                    test_z = test_z_scaling

            test_objs.append(objs)
            if num_ineq != 0:
                test_ineq_vio_maxs.append(ineq_vio_maxs)
                test_ineq_vio_means.append(ineq_vio_means)
            if num_eq != 0:
                test_eq_vio_maxs.append(eq_vio_maxs)
                test_eq_vio_means.append(eq_vio_means)
            if num_lb != 0:
                test_lb_vio_maxs.append(lb_vio_maxs)
                test_lb_vio_means.append(lb_vio_means)
            if num_ub != 0:
                test_ub_vio_maxs.append(ub_vio_maxs)
                test_ub_vio_means.append(ub_vio_means)
            test_ls_residuals.append(ls_residuals)
            test_primal_residuals.append(primal_residuals)
            test_dual_residuals.append(dual_residuals)
            test_x_cond_1_left.append(x_cond_1_left)
            test_x_cond_1_right.append(x_cond_1_right)
            test_x_cond_2_left.append(x_cond_2_left)
            test_x_cond_2_right.append(x_cond_2_right)
            test_z_cond_1_left.append(z_cond_1_left)
            test_z_cond_1_right.append(z_cond_1_right)
            test_z_cond_2_left.append(z_cond_2_left)
            test_z_cond_2_right.append(z_cond_2_right)
            test_alpha_cond_left.append(alpha_cond_left)
            test_alpha_cond_right.append(alpha_cond_right)

            if args.scaling:
                test_Q = test_Q_pre
                test_p = test_p_pre
                test_A0 = test_A0_pre
                test_zl = test_zl_pre
                test_zu = test_zu_pre

                start_time = time.time()
                test_x = torch.bmm(scaling.D, test_x)
                test_y = torch.bmm(scaling.cinv * scaling.E, test_y)
                test_z = torch.bmm(scaling.Einv, test_z)
                end_time = time.time()

                scaling_time += (end_time-start_time)
                test_total_time += scaling_time


            ## Stage II
            if args.feas_rest:
                # save results
                objs_fr = []
                if num_ineq != 0:
                    ineq_vio_maxs_fr = []
                    ineq_vio_means_fr = []
                if num_eq != 0:
                    eq_vio_maxs_fr = []
                    eq_vio_means_fr = []
                if num_lb != 0:
                    lb_vio_maxs_fr = []
                    lb_vio_means_fr = []
                if num_ub != 0:
                    ub_vio_maxs_fr = []
                    ub_vio_means_fr = []
                test_ls_residuals_fr = []
                primal_residuals_fr = []
                dual_residuals_fr = []

                for t in range(args.feas_rest_num):
                    start_time = time.time()
                    if t == 0:
                        test_lu = None
                        test_A_tild = None
                        test_piv = None

                    test_x, test_y, test_z, test_xv, test_A_tild, test_b_tild, test_lu, test_piv = exact_model(test_rho_vec, test_x, test_y, test_z, test_xv,
                                                                                               test_sigma, test_A_tild, test_lu, test_piv,
                                                                                               Q=test_Q, p=test_p, A0=test_A0, lb=test_lb, ub=test_ub,
                                                                                               zl=test_zl, zu=test_zu)
                    end_time = time.time()
                    test_total_time += (end_time - start_time)

                    # obj
                    test_obj = obj_fn(test_x, Q=test_Q, p=test_p).mean()
                    objs_fr.append(test_obj.detach().cpu().numpy())
                    # linear system residues
                    test_ls_res = torch.linalg.vector_norm(torch.bmm(test_A_tild, test_xv) - test_b_tild, dim=(1, 2), keepdim=True).mean()
                    test_ls_residuals_fr.append(test_ls_res.detach().cpu().numpy())
                    # primal and dual residues
                    test_primal_res, test_dual_res, loss = primal_dual_loss(test_x, test_y, test_z, test_Q, test_p, test_A0)
                    primal_residuals_fr.append(test_primal_res.mean().detach().cpu().numpy())
                    dual_residuals_fr.append(test_dual_res.mean().detach().cpu().numpy())


                    if num_ineq != 0:
                        test_ineq_vio_max = ineq_dist(test_x, G=test_G, c=test_c).max(dim=1).values.mean()
                        test_ineq_vio_mean = ineq_dist(test_x, G=test_G, c=test_c).mean()
                        ineq_vio_maxs_fr.append(test_ineq_vio_max.detach().cpu().numpy())
                        ineq_vio_means_fr.append(test_ineq_vio_mean.detach().cpu().numpy())
                    if num_eq != 0:
                        test_eq_vio_max = eq_dist(test_x, A=test_A, b=test_b).max(dim=1).values.mean()
                        test_eq_vio_mean = eq_dist(test_x, A=test_A, b=test_b).mean()
                        eq_vio_maxs_fr.append(test_eq_vio_max.detach().cpu().numpy())
                        eq_vio_means_fr.append(test_eq_vio_mean.detach().cpu().numpy())
                    if num_lb != 0:
                        test_lb_vio_max = lb_dist(test_x, lb=test_lb).max(dim=1).values.mean()
                        test_lb_vio_mean = lb_dist(test_x, lb=test_lb).mean()
                        lb_vio_maxs_fr.append(test_lb_vio_max.detach().cpu().numpy())
                        lb_vio_means_fr.append(test_lb_vio_mean.detach().cpu().numpy())
                    if num_ub != 0:
                        test_ub_vio_max = ub_dist(test_x, ub=test_ub).max(dim=1).values.mean()
                        test_ub_vio_mean = ub_dist(test_x, ub=test_ub).mean()
                        ub_vio_maxs_fr.append(test_ub_vio_max.detach().cpu().numpy())
                        ub_vio_means_fr.append(test_ub_vio_mean.detach().cpu().numpy())

                    test_objs_fr.append(objs_fr)
                    if num_ineq != 0:
                        test_ineq_vio_maxs_fr.append(ineq_vio_maxs_fr)
                        test_ineq_vio_means_fr.append(ineq_vio_means_fr)
                    if num_eq != 0:
                        test_eq_vio_maxs_fr.append(eq_vio_maxs_fr)
                        test_eq_vio_means_fr.append(eq_vio_means_fr)
                    if num_lb != 0:
                        test_lb_vio_maxs_fr.append(lb_vio_maxs_fr)
                        test_lb_vio_means_fr.append(lb_vio_means_fr)
                    if num_ub != 0:
                        test_ub_vio_maxs_fr.append(ub_vio_maxs_fr)
                        test_ub_vio_means_fr.append(ub_vio_means_fr)
                    test_primal_residuals_fr.append(primal_residuals_fr)
                    test_dual_residuals_fr.append(dual_residuals_fr)

        for t in range(args.test_outer_T):
            test_obj = np.array(test_objs).mean(axis=0)[t]
            test_pri_res = np.array(test_primal_residuals).mean(axis=0)[t]
            test_dual_res = np.array(test_dual_residuals).mean(axis=0)[t]
            print("Epoch : {} | Test_Obj : {:.3f}".format(t, test_obj))
            print("Primal_Residuals : {} | Dual_Residuals : {}".format(test_pri_res, test_dual_res))
            if num_ineq != 0:
                test_ineq_vio_max = np.array(test_ineq_vio_maxs).mean(axis=0)[t]
                test_ineq_vio_mean = np.array(test_ineq_vio_means).mean(axis=0)[t]
                print("Test_Max_Ineq : {:.3f} | Test_Mean_Ineq : {:.3f} |".format(test_ineq_vio_max, test_ineq_vio_mean))
            if num_eq != 0:
                test_eq_vio_max = np.array(test_eq_vio_maxs).mean(axis=0)[t]
                test_eq_vio_mean = np.array(test_eq_vio_means).mean(axis=0)[t]
                print("Test_Max_Eq : {:.3f} | Test_Mean_Eq : {:.3f} |".format(test_eq_vio_max, test_eq_vio_mean))
            if num_lb != 0:
                test_lb_vio_max = np.array(test_lb_vio_maxs).mean(axis=0)[t]
                test_lb_vio_mean = np.array(test_lb_vio_means).mean(axis=0)[t]
                print("Test_Max_Lb : {:.3f} | Test_Mean_Lb : {:.3f} |".format(test_lb_vio_max, test_lb_vio_mean))
            if num_ub != 0:
                test_ub_vio_max = np.array(test_ub_vio_maxs).mean(axis=0)[t]
                test_ub_vio_mean = np.array(test_ub_vio_means).mean(axis=0)[t]
                print("Test_Max_Ub : {:.3f} | Test_Mean_Ub : {:.3f} |".format(test_ub_vio_max, test_ub_vio_mean))

        if args.feas_rest:
            print("-----Starting Sage II-----")
            for t in range(args.feas_rest_num):
                test_obj = np.array(test_objs_fr).mean(axis=0)[t]
                print("Epoch : {} | Test_Obj : {:.3f}".format(t, test_obj))
                # print("Primal_Residuals : {} | Dual_Residuals : {}".format(test_pri_res, test_dual_res))
                if num_ineq != 0:
                    test_ineq_vio_max = np.array(test_ineq_vio_maxs_fr).mean(axis=0)[t]
                    test_ineq_vio_mean = np.array(test_ineq_vio_means_fr).mean(axis=0)[t]
                    print("Test_Max_Ineq : {:.3f} | Test_Mean_Ineq : {:.3f} |".format(test_ineq_vio_max, test_ineq_vio_mean))
                if num_eq != 0:
                    test_eq_vio_max = np.array(test_eq_vio_maxs_fr).mean(axis=0)[t]
                    test_eq_vio_mean = np.array(test_eq_vio_means_fr).mean(axis=0)[t]
                    print("Test_Max_Eq : {:.3f} | Test_Mean_Eq : {:.3f} |".format(test_eq_vio_max, test_eq_vio_mean))
                if num_lb != 0:
                    test_lb_vio_max = np.array(test_lb_vio_maxs_fr).mean(axis=0)[t]
                    test_lb_vio_mean = np.array(test_lb_vio_means_fr).mean(axis=0)[t]
                    print("Test_Max_Lb : {:.3f} | Test_Mean_Lb : {:.3f} |".format(test_lb_vio_max, test_lb_vio_mean))
                if num_ub != 0:
                    test_ub_vio_max = np.array(test_ub_vio_maxs_fr).mean(axis=0)[t]
                    test_ub_vio_mean = np.array(test_ub_vio_means_fr).mean(axis=0)[t]
                    print("Test_Max_Ub : {:.3f} | Test_Mean_Ub : {:.3f} |".format(test_ub_vio_max, test_ub_vio_mean))
        print("Parallel Time : {}".format((test_total_time) / test_size))

        if args.save_sol:
            if args.prob_type == 'QP_RHS':
                results_save_path = os.path.join(args.save_dir, model.name(),
                                         'QP_RHS_{}_{}_{}_{}_{}_results.mat'.format(args.num_var,
                                                                            args.num_eq,
                                                                            args.num_ineq,
                                                                            args.outer_T,
                                                                            args.hidden_dim))
            elif args.prob_type == 'QP':
                results_save_path = os.path.join(args.save_dir, model.name(),
                                                 'QP_{}_{}_{}_{}_{}_results.mat'.format(args.num_var,
                                                                                            args.num_eq,
                                                                                            args.num_ineq,
                                                                                            args.outer_T,
                                                                                            args.hidden_dim))
            elif args.prob_type == 'Random_QP':
                results_save_path = os.path.join(args.save_dir, model.name(), 'Random_QP_{}_{}_{}_{}_results.mat'.format(args.num_var,
                                                                            args.num_ineq,
                                                                            args.outer_T,
                                                                            args.hidden_dim))
            elif args.prob_type == 'Equality_QP':
                results_save_path = os.path.join(args.save_dir, model.name(), 'Equality_QP_{}_{}_{}_{}_results.mat'.format(args.num_var,
                                                                                                                                args.num_eq,
                                                                                                                                args.outer_T,
                                                                                                                                args.hidden_dim))
            elif args.prob_type == 'SVM':
                results_save_path = os.path.join(args.save_dir, model.name(), 'SVM_{}_{}_{}_{}.pth'.format(args.num_var,
                                                                      args.num_ineq,
                                                                      args.outer_T,
                                                                      args.hidden_dim))
            elif args.prob_type == 'QPLIB':
                results_save_path = os.path.join(args.save_dir, model.name(), '{}_{}_{}_{}_results.mat'.format(args.prob_type,
                                                                                                                    args.qplib_num,
                                                                                                                    args.outer_T,
                                                                                                                    args.hidden_dim))
            elif args.prob_type == 'MM_MOSARQP2':
                results_save_path = os.path.join(args.save_dir, model.name(), '{}_{}_{}.pth'.format(args.prob_type,
                                                                                                      args.outer_T,
                                                                                                      args.hidden_dim))

            elif args.prob_type == 'MM_QSHIP04L':
                # model parameter save path
                results_save_path = os.path.join(args.save_dir, model.name(), '{}_{}_{}_results.mat'.format(args.prob_type,
                                                                                                      args.outer_T,
                                                                                                      args.hidden_dim))
            elif args.prob_type == 'MM_QSHIP08S':
                # model parameter save path
                results_save_path = os.path.join(args.save_dir, model.name(), '{}_{}_{}_results.mat'.format(args.prob_type,
                                                                                                      args.outer_T,
                                                                                                      args.hidden_dim))
            elif args.prob_type == 'MM_CVXQP1_M':
                # model parameter save path
                results_save_path = os.path.join(args.save_dir, model.name(), '{}_{}_{}_results.mat'.format(args.prob_type,
                                                                                                      args.outer_T,
                                                                                                      args.hidden_dim))
            elif args.prob_type == 'MM_CVXQP3_M':
                # model parameter save path
                results_save_path = os.path.join(args.save_dir, model.name(), '{}_{}_{}_results.mat'.format(args.prob_type,
                                                                                                      args.outer_T,
                                                                                                      args.hidden_dim))


            if args.feas_rest:
                test_dict = {'time': (test_total_time),
                         'x': np.array(test_x.detach().cpu()),
                         'objs': np.array(test_objs),
                         'ls_res': np.array(test_ls_residuals),
                         'primal_res': np.array(test_primal_residuals),
                         'dual_res': np.array(test_dual_residuals),
                         'objs_fr': np.array(test_objs_fr),
                         'ls_res_fr': np.array(test_ls_residuals_fr),
                         'primal_res_fr': np.array(test_primal_residuals_fr),
                         'dual_res_fr': np.array(test_dual_residuals_fr),
                         'x_cond_1_left': np.array(test_x_cond_1_left),
                         'x_cond_1_right': np.array(test_x_cond_1_right),
                         'x_cond_2_left': np.array(test_x_cond_2_left),
                         'x_cond_2_right': np.array(test_x_cond_2_right),
                         'z_cond_1_left': np.array(test_z_cond_1_left),
                         'z_cond_1_right': np.array(test_z_cond_1_right),
                         'z_cond_2_left': np.array(test_z_cond_2_left),
                         'z_cond_2_right': np.array(test_z_cond_2_right),
                         'alpha_cond_left': np.array(test_alpha_cond_left),
                         'alpha_cond_right': np.array(test_alpha_cond_right),
                        }
            else:
                test_dict = {'time': (test_total_time),
                             'x': np.array(test_x.detach().cpu()),
                             'objs': np.array(test_objs),
                             'ls_res': np.array(test_ls_residuals),
                             'primal_res': np.array(test_primal_residuals),
                             'dual_res': np.array(test_dual_residuals),
                             'x_cond_1_left': np.array(test_x_cond_1_left),
                             'x_cond_1_right': np.array(test_x_cond_1_right),
                             'x_cond_2_left': np.array(test_x_cond_2_left),
                             'x_cond_2_right': np.array(test_x_cond_2_right),
                             'z_cond_1_left': np.array(test_z_cond_1_left),
                             'z_cond_1_right': np.array(test_z_cond_1_right),
                             'z_cond_2_left': np.array(test_z_cond_2_left),
                             'z_cond_2_right': np.array(test_z_cond_2_right),
                             'alpha_cond_left': np.array(test_alpha_cond_left),
                             'alpha_cond_right': np.array(test_alpha_cond_right),
                             }

            # save test results
            sio.savemat(results_save_path, test_dict)

