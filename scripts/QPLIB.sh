---

#QPLIB_8845
python main.py --config .\configs\QP.yaml --model_name LSTM --prob_type QPLIB --qplib_num 8845 --outer_T 300 --truncated_length 150 --hidden_dim 300 --eq_tol 50 --ineq_tol 50 --batch_size 2 --input_dim 2 --data_size 1000 --lr 0.00005  --early_stop_mode max --scaling
python main.py --config .\configs\QP.yaml --model_name LSTM --prob_type QPLIB --qplib_num 8845 --outer_T 300 --truncated_length 150 --hidden_dim 300 --eq_tol 50 --ineq_tol 50 --batch_size 2 --input_dim 2 --data_size 1000 --lr 0.00005  --early_stop_mode max --scaling --test --test_outer_T 300 --test_batch_size 100 --test_frac 0.1


#QPLIB_8906
python main.py --config .\configs\QP.yaml --model_name LSTM --prob_type QPLIB --qplib_num 8906 --outer_T 300 --truncated_length 50 --hidden_dim 200 --eq_tol 1 --ineq_tol 1 --batch_size 1 --input_dim 2 --data_size 1000 --lr 0.00005  --early_stop_mode max --scaling --val_frac 0.002
python main.py --config .\configs\QP.yaml --model_name LSTM --prob_type QPLIB --qplib_num 8906 --outer_T 300 --truncated_length 50 --hidden_dim 200 --eq_tol 1 --ineq_tol 1 --batch_size 1 --input_dim 2 --data_size 1000 --lr 0.00005  --early_stop_mode max --scaling --val_frac 0.002 --test_frac 0.01 --test --test_outer_T 300 --feas_rest
