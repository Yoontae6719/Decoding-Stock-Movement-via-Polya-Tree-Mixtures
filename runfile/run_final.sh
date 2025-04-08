# run_final.sh
export CUDA_VISIBLE_DEVICES=0

model_name=treeMoE

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path BNTX.feather \
  --model_id BNTX \
  --model $model_name \
  --data BNTX \
  --dim_input 437 \
  --alpha_fs 0.9 \
  --beta_fs 1.0 \
  --max_depth 2 \
  --hidden_dim_expert 16 \
  --lambda_KL 0.01 \
  --use_leaf_feature_selector_only 1 \
  --use_gating_mlp 1 \
  --gating_mlp_hidden 16 \
  --lradj cosine \
  --anneal_epochs 5 \
  --schedule_type linear \
  --learning_rate 0.0003 \
  --des 'EXPNEW' \
  --itr 3 &  # Fixed itr value since we're not using it as a parameter


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path AAPL.feather \
  --model_id AAPL \
  --model $model_name \
  --data AAPL \
  --dim_input 437 \
  --alpha_fs 0.5 \
  --beta_fs 1.0 \
  --max_depth 3 \
  --hidden_dim_expert 32 \
  --lambda_KL 0.01 \
  --use_leaf_feature_selector_only 1 \
  --use_gating_mlp 1 \
  --gating_mlp_hidden 8 \
  --lradj cosine \
  --anneal_epochs 5 \
  --schedule_type linear \
  --learning_rate 0.0003 \
  --des 'EXPNEW' \
  --itr 3 &  # Fixed itr value since we're not using it as a parameter


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path AMZN.feather \
  --model_id AMZN \
  --model $model_name \
  --data AMZN \
  --dim_input 437 \
  --alpha_fs 0.9 \
  --beta_fs 1.0 \
  --max_depth 4 \
  --hidden_dim_expert 32 \
  --lambda_KL 0.01 \
  --use_leaf_feature_selector_only 0 \
  --use_gating_mlp 1 \
  --gating_mlp_hidden 16 \
  --lradj cosine \
  --anneal_epochs 5 \
  --schedule_type linear \
  --learning_rate 0.0003 \
  --des 'EXPNEW' \
  --itr 3 &  # Fixed itr value since we're not using it as a parameter

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path JPM.feather \
  --model_id JPM \
  --model $model_name \
  --data JPM \
  --dim_input 437 \
  --alpha_fs 0.9 \
  --beta_fs 1.0 \
  --max_depth 2 \
  --hidden_dim_expert 32 \
  --lambda_KL 0.01 \
  --use_leaf_feature_selector_only 0 \
  --use_gating_mlp 1 \
  --gating_mlp_hidden 16 \
  --lradj cosine \
  --anneal_epochs 5 \
  --schedule_type linear \
  --learning_rate 0.0003 \
  --des 'EXPNEW' \
  --itr 3 &  # Fixed itr value since we're not using it as a parameter

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path META.feather \
  --model_id META \
  --model $model_name \
  --data META \
  --dim_input 437 \
  --alpha_fs 0.9 \
  --beta_fs 1.0 \
  --max_depth 2 \
  --hidden_dim_expert 16 \
  --lambda_KL 0.01 \
  --use_leaf_feature_selector_only 0 \
  --use_gating_mlp 0 \
  --gating_mlp_hidden 32 \
  --lradj cosine \
  --anneal_epochs 5 \
  --schedule_type linear \
  --learning_rate 0.0003 \
  --des 'EXPNEW' \
  --itr 3 &  # Fixed itr value since we're not using it as a parameter


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path NFLX.feather \
  --model_id NFLX \
  --model $model_name \
  --data NFLX \
  --dim_input 437 \
  --alpha_fs 0.7 \
  --beta_fs 1.0 \
  --max_depth 3 \
  --hidden_dim_expert 16 \
  --lambda_KL 0.01 \
  --use_leaf_feature_selector_only 1 \
  --use_gating_mlp 1 \
  --gating_mlp_hidden 8 \
  --lradj cosine \
  --anneal_epochs 5 \
  --schedule_type linear \
  --learning_rate 0.0003 \
  --des 'EXPNEW' \
  --itr 3 &  # Fixed itr value since we're not using it as a parameter

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path NVDA.feather \
  --model_id NVDA \
  --model $model_name \
  --data NVDA \
  --dim_input 437 \
  --alpha_fs 0.7 \
  --beta_fs 1.0 \
  --max_depth 4 \
  --hidden_dim_expert 8 \
  --lambda_KL 0.01 \
  --use_leaf_feature_selector_only 0 \
  --use_gating_mlp 1 \
  --gating_mlp_hidden 16 \
  --lradj cosine \
  --anneal_epochs 5 \
  --schedule_type linear \
  --learning_rate 0.0003 \
  --des 'EXPNEW' \
  --itr 3 &  # Fixed itr value since we're not using it as a parameter

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path PEP.feather \
  --model_id PEP \
  --model $model_name \
  --data PEP \
  --dim_input 437 \
  --alpha_fs 0.9 \
  --beta_fs 1.0 \
  --max_depth 2 \
  --hidden_dim_expert 16 \
  --lambda_KL 0.01 \
  --use_leaf_feature_selector_only 1 \
  --use_gating_mlp 1 \
  --gating_mlp_hidden 8 \
  --lradj cosine \
  --anneal_epochs 5 \
  --schedule_type linear \
  --learning_rate 0.0003 \
  --des 'EXPNEW' \
  --itr 3 &  # Fixed itr value since we're not using it as a parameter

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path TSLA.feather \
  --model_id TSLA \
  --model $model_name \
  --data TSLA \
  --dim_input 437 \
  --alpha_fs 0.9 \
  --beta_fs 1.0 \
  --max_depth 2 \
  --hidden_dim_expert 32 \
  --lambda_KL 0.01 \
  --use_leaf_feature_selector_only 0 \
  --use_gating_mlp 0 \
  --gating_mlp_hidden 32 \
  --lradj cosine \
  --anneal_epochs 5 \
  --schedule_type linear \
  --learning_rate 0.0003 \
  --des 'EXPNEW' \
  --itr 3 &  # Fixed itr value since we're not using it as a parameter
