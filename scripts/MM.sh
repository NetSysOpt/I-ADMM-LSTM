---

# MM_MOSARQP2
python main.py --config ./configs/QP.yaml --model_name LSTM --prob_type MM_MOSARQP2 --outer_T 300 --truncated_length 100 --hidden_dim 400 --eq_tol 0.002 --ineq_tol 0.002 --batch_size 1 --input_dim 2 --data_size 1000 --lr 0.00005  --early_stop_mode min --scaling
python main.py --config ./configs/QP.yaml --model_name LSTM --prob_type MM_MOSARQP2 --outer_T 300 --truncated_length 100 --hidden_dim 400 --eq_tol 0.002 --ineq_tol 0.002 --batch_size 1 --input_dim 2 --data_size 1000 --lr 0.00005  --early_stop_mode min --scaling --test --test_outer_T 300 --save_sol

# MM_QSCRS8
python main.py --config ./configs/QP.yaml --model_name LSTM --prob_type MM_QSCRS8 --outer_T 300 --truncated_length 100 --hidden_dim 400 --eq_tol 1 --ineq_tol 1 --batch_size 1 --input_dim 2 --data_size 1000 --lr 0.00005  --early_stop_mode max --scaling
python main.py --config ./configs/QP.yaml --model_name LSTM --prob_type MM_QSCRS8 --outer_T 300 --truncated_length 100 --hidden_dim 400 --eq_tol 1 --ineq_tol 1 --batch_size 1 --input_dim 2 --data_size 1000 --lr 0.00005  --early_stop_mode max --scaling --test --test_outer_T 300
python main.py --config ./configs/QP.yaml --model_name LSTM --prob_type MM_QSCRS8 --outer_T 300 --truncated_length 100 --hidden_dim 400 --eq_tol 1 --ineq_tol 1 --batch_size 1 --input_dim 2 --data_size 1000 --lr 0.00005  --early_stop_mode max --scaling --test --test_outer_T 300 --feas_res

# MM_QSCSD6
python main.py --config ./configs/QP.yaml --model_name LSTM --prob_type MM_QSCSD6 --outer_T 50 --truncated_length 50 --hidden_dim 800 --eq_tol 1 --ineq_tol 1 --batch_size 2 --input_dim 2 --data_size 1000 --lr 0.00005  --early_stop_mode max --scaling


# MM_Q25FV47
python main.py --config ./configs/QP.yaml --model_name LSTM --prob_type MM_Q25FV47 --outer_T 450 --truncated_length 150 --hidden_dim 300 --eq_tol 20 --ineq_tol 20 --batch_size 2 --input_dim 2 --data_size 1000 --lr 0.00005  --early_stop_mode max --scaling
python main.py --config ./configs/QP.yaml --model_name LSTM --prob_type MM_Q25FV47 --outer_T 450 --truncated_length 150 --hidden_dim 300 --eq_tol 20 --ineq_tol 20 --batch_size 2 --input_dim 2 --data_size 1000 --lr 0.00005  --early_stop_mode max --scaling --test --test_outer_T 450


# MM_QSHIP04L
python main.py --config ./configs/QP.yaml --model_name LSTM --prob_type MM_QSHIP04L --outer_T 200 --truncated_length 100 --hidden_dim 400 --eq_tol 1 --ineq_tol 1 --batch_size 2 --input_dim 2 --data_size 1000 --lr 0.00005 --scaling  --early_stop_mode max
python main.py --config ./configs/QP.yaml --model_name LSTM --prob_type MM_QSHIP04L --outer_T 200 --truncated_length 100 --hidden_dim 400 --eq_tol 1 --ineq_tol 1 --batch_size 2 --input_dim 2 --data_size 1000 --lr 0.00005 --scaling  --early_stop_mode max --test --test_outer_T 200


# MM_QSHIP08S
python main.py --config ./configs/QP.yaml --model_name LSTM --prob_type MM_QSHIP08S --outer_T 200 --truncated_length 100 --hidden_dim 400 --eq_tol 1 --ineq_tol 1 --batch_size 1 --input_dim 2 --data_size 1000 --lr 0.00005 --scaling  --early_stop_mode max
python main.py --config ./configs/QP.yaml --model_name LSTM --prob_type MM_QSHIP08S --outer_T 200 --truncated_length 100 --hidden_dim 400 --eq_tol 1 --ineq_tol 1 --batch_size 1 --input_dim 2 --data_size 1000 --lr 0.00005 --scaling  --early_stop_mode max --test --test_outer_T 200 --test_batch_size 100 --test_frac 0.1
python main.py --config ./configs/QP.yaml --model_name LSTM --prob_type MM_QSHIP08S --outer_T 200 --truncated_length 100 --hidden_dim 400 --eq_tol 1 --ineq_tol 1 --batch_size 1 --input_dim 2 --data_size 1000 --lr 0.00005 --scaling  --early_stop_mode max --test --test_outer_T 200 --save_sol


# MM_CVXQP1_M
python main.py --config ./configs/QP.yaml --model_name LSTM --prob_type MM_CVXQP1_M --outer_T 50 --truncated_length 50 --hidden_dim 800 --eq_tol 5 --ineq_tol 5 --batch_size 2 --input_dim 2 --data_size 1000 --lr 0.00005 --scaling  --early_stop_mode max
python main.py --config ./configs/QP.yaml --model_name LSTM --prob_type MM_CVXQP1_M --outer_T 50 --truncated_length 50 --hidden_dim 800 --eq_tol 5 --ineq_tol 5 --batch_size 2 --input_dim 2 --data_size 1000 --lr 0.00005 --scaling  --early_stop_mode max --test --test_outer_T 50 --test_batch_size 1
python main.py --config ./configs/QP.yaml --model_name LSTM --prob_type MM_CVXQP1_M --outer_T 50 --truncated_length 50 --hidden_dim 800 --eq_tol 5 --ineq_tol 5 --batch_size 2 --input_dim 2 --data_size 1000 --lr 0.00005 --scaling  --early_stop_mode max --test --test_outer_T 50 --feas_rest


# MM_CVXQP3_M
python main.py --config .\configs\QP.yaml --model_name LSTM --prob_type MM_CVXQP3_M --outer_T 50 --truncated_length 50 --hidden_dim 800 --eq_tol 1 --ineq_tol 1 --batch_size 2 --input_dim 2 --data_size 1000 --lr 0.00005 --scaling  --early_stop_mode max
python main.py --config .\configs\QP.yaml --model_name LSTM --prob_type MM_CVXQP3_M --outer_T 50 --truncated_length 50 --hidden_dim 800 --eq_tol 1 --ineq_tol 1 --batch_size 2 --input_dim 2 --data_size 1000 --lr 0.00005 --scaling  --early_stop_mode max --test --test_outer_T 50
