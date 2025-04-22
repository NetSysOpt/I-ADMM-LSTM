---
#QP_1000_500_500
python main.py --config .\configs\QP.yaml --model_name LSTM --prob_type QP --outer_T 100 --truncated_length 100 --hidden_dim 800 --eq_tol 0.2 --ineq_tol 0.2 --num_var 1000 --num_ineq 500 --num_eq 500 --input_dim 2 --data_size 1000 --batch_size 2 --lr 0.00005 --scaling
python main.py --config .\configs\QP.yaml --model_name LSTM --prob_type QP --outer_T 100 --truncated_length 100 --hidden_dim 800 --eq_tol 0.2 --ineq_tol 0.2 --num_var 1000 --num_ineq 500 --num_eq 500 --input_dim 2 --data_size 1000 --batch_size 2 --lr 0.00005 --scaling --test --test_outer_T 100 --test_batch_size 1 --test_frac 0.1
python main.py --config .\configs\QP.yaml --model_name LSTM --prob_type QP --outer_T 100 --truncated_length 100 --hidden_dim 800 --eq_tol 0.2 --ineq_tol 0.2 --num_var 1000 --num_ineq 500 --num_eq 500 --input_dim 2 --data_size 1000 --batch_size 2 --lr 0.00005 --scaling --test --test_outer_T 100 --test_batch_size 1 --save_sol


#QP_1500_750_750
python main.py --config .\configs\QP.yaml --model_name LSTM --prob_type QP --outer_T 100 --truncated_length 100 --hidden_dim 800 --eq_tol 0.2 --ineq_tol 0.2 --num_var 1500 --num_ineq 750 --num_eq 750 --input_dim 2 --data_size 1000 --batch_size 2 --lr 0.00005 --scaling
python main.py --config .\configs\QP.yaml --model_name LSTM --prob_type QP --outer_T 100 --truncated_length 100 --hidden_dim 800 --eq_tol 0.2 --ineq_tol 0.2 --num_var 1500 --num_ineq 750 --num_eq 750 --input_dim 2 --data_size 1000 --batch_size 2 --lr 0.00005 --scaling --test --test_outer_T 100
python main.py --config .\configs\QP.yaml --model_name LSTM --prob_type QP --outer_T 100 --truncated_length 100 --hidden_dim 800 --eq_tol 0.2 --ineq_tol 0.2 --num_var 1500 --num_ineq 750 --num_eq 750 --input_dim 2 --data_size 1000 --batch_size 2 --lr 0.00005 --scaling --test --test_outer_T 100 --feas_rest --save_sol

#QP_RHS_1000_500_500
python main.py --config .\configs\QP.yaml --model_name LSTM --prob_type QP_RHS --outer_T 100 --truncated_length 100 --hidden_dim 400 --eq_tol 0.2 --ineq_tol 0.2 --num_var 1000 --num_ineq 500 --num_eq 500 --input_dim 2 --data_size 1000 --batch_size 2 --lr 0.00005 --scaling
python main.py --config .\configs\QP.yaml --model_name LSTM --prob_type QP_RHS --outer_T 100 --truncated_length 100 --hidden_dim 400 --eq_tol 0.2 --ineq_tol 0.2 --num_var 1000 --num_ineq 500 --num_eq 500 --input_dim 2 --data_size 1000 --batch_size 2 --lr 0.00005 --scaling --test --test_outer_T 100
python main.py --config .\configs\QP.yaml --model_name LSTM --prob_type QP_RHS --outer_T 100 --truncated_length 100 --hidden_dim 400 --eq_tol 0.2 --ineq_tol 0.2 --num_var 1000 --num_ineq 500 --num_eq 500 --input_dim 2 --data_size 1000 --batch_size 2 --lr 0.00005 --scaling --test --test_outer_T 100 --feas_rest --test_batch_size 1 --save_sol

#QP_RHS_1500_750_750
python main.py --config .\configs\QP.yaml --model_name LSTM --prob_type QP_RHS --outer_T 150 --truncated_length 150 --hidden_dim 400 --eq_tol 0.2 --ineq_tol 0.2 --num_var 1500 --num_ineq 750 --num_eq 750 --input_dim 2 --data_size 1000 --batch_size 2 --lr 0.00005 --scaling
python main.py --config .\configs\QP.yaml --model_name LSTM --prob_type QP_RHS --outer_T 150 --truncated_length 150 --hidden_dim 400 --eq_tol 0.2 --ineq_tol 0.2 --num_var 1500 --num_ineq 750 --num_eq 750 --input_dim 2 --data_size 1000 --batch_size 2 --lr 0.00005 --scaling --test --test_outer_T 150 --test_batch_size 1
python main.py --config .\configs\QP.yaml --model_name LSTM --prob_type QP_RHS --outer_T 150 --truncated_length 150 --hidden_dim 400 --eq_tol 0.2 --ineq_tol 0.2 --num_var 1500 --num_ineq 750 --num_eq 750 --input_dim 2 --data_size 1000 --batch_size 2 --lr 0.00005 --scaling --test --test_outer_T 150 --feas_rest --save_sol

#Equality_QP_1000_500
python main.py --config .\configs\QP.yaml --model_name LSTM --prob_type Equality_QP --outer_T 400 --truncated_length 200 --hidden_dim 200 --eq_tol 0.5 --ineq_tol 0.5 --num_var 1000 --num_eq 500 --input_dim 2 --data_size 1000 --batch_size 2 --lr 0.00005 --scaling
python main.py --config .\configs\QP.yaml --model_name LSTM --prob_type Equality_QP --outer_T 400 --truncated_length 200 --hidden_dim 200 --eq_tol 0.5 --ineq_tol 0.5 --num_var 1000 --num_eq 500 --input_dim 2 --data_size 1000 --batch_size 2 --lr 0.00005 --scaling --test --test_outer_T 400 --save_sol
python main.py --config .\configs\QP.yaml --model_name LSTM --prob_type Equality_QP --outer_T 400 --truncated_length 200 --hidden_dim 200 --eq_tol 0.5 --ineq_tol 0.5 --num_var 1000 --num_eq 500 --input_dim 2 --data_size 1000 --batch_size 2 --lr 0.00005 --scaling --test --test_outer_T 400 --save_sol

#Random_QP_1000_2000
python main.py --config .\configs\QP.yaml --model_name LSTM --prob_type Random_QP --outer_T 600 --truncated_length 150 --hidden_dim 200 --eq_tol 1 --ineq_tol 1 --num_var 1000 --num_ineq 2000 --input_dim 2 --data_size 1000 --batch_size 2 --lr 0.00005 --scaling
python main.py --config .\configs\QP.yaml --model_name LSTM --prob_type Random_QP --outer_T 600 --truncated_length 150 --hidden_dim 200 --eq_tol 1 --ineq_tol 1 --num_var 1000 --num_ineq 2000 --input_dim 2 --data_size 1000 --batch_size 2 --lr 0.00005 --scaling --test --test_outer_T 600 --test_batch_size 100 --test_frac 0.1
python main.py --config .\configs\QP.yaml --model_name LSTM --prob_type Random_QP --outer_T 600 --truncated_length 150 --hidden_dim 200 --eq_tol 1 --ineq_tol 1 --num_var 1000 --num_ineq 2000 --input_dim 2 --data_size 1000 --batch_size 2 --lr 0.00005 --scaling --test --test_outer_T 600 --save_sol


# SVM
python main.py --config .\configs\QP.yaml --model_name LSTM --prob_type SVM --outer_T 50 --truncated_length 50 --hidden_dim 800 --eq_tol 0.01 --ineq_tol 0.01 --num_var 1500 --num_ineq 500 --input_dim 2 --data_size 1000 --batch_size 2 --lr 0.00005 --early_stop_mode min --scaling
python main.py --config .\configs\QP.yaml --model_name LSTM --prob_type SVM --outer_T 50 --truncated_length 50 --hidden_dim 800 --eq_tol 0.01 --ineq_tol 0.01 --num_var 1500 --num_ineq 500 --input_dim 2 --data_size 1000 --batch_size 2 --lr 0.00005 --early_stop_mode min --scaling --test --test_outer_T 50 --test_batch_size 2 --test_frac 0.1
python main.py --config .\configs\QP.yaml --model_name LSTM --prob_type SVM --outer_T 50 --truncated_length 50 --hidden_dim 800 --eq_tol 0.01 --ineq_tol 0.01 --num_var 1500 --num_ineq 500 --input_dim 2 --data_size 1000 --batch_size 2 --lr 0.00005 --early_stop_mode min --scaling --test --test_outer_T 50 --save_sol
