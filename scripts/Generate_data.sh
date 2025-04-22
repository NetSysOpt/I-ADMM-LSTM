---

# Random_QP_1000_2000
python generate_data.py --config .\configs\Generate_Data.yaml --prob_type Random_QP --num_var 1000 --num_ineq 2000

# Equality_QP_1000_500
python generate_data.py --config .\configs\Generate_Data.yaml --prob_type Equality_QP --num_var 1000 --num_eq 500

# SVM_1000_500
python generate_data.py --config .\configs\Generate_Data.yaml --prob_type Equality_QP --num_var 1000 --num_eq 500


# MM_Q25FV47
python generate_data.py --config .\configs\Generate_Data.yaml --prob_type MM_Q25FV47

# MM_QSHIP04L
python generate_data.py --config .\configs\Generate_Data.yaml --prob_type MM_QSHIP04L