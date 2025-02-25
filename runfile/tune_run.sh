export CUDA_VISIBLE_DEVICES=0

model_name=treeMoE


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path FULL.feather \
  --model_id FULL_run \
  --model $model_name \
  --data FULL \
  --dim_input 437 \
  --max_depth 5 \
  --hidden_dim_expert 16 \
  --alpha_fs 1.2241694728131247 \
  --beta_fs 0.28837943783729975 \
  --use_gating_mlp 1 \
  --gating_mlp_hidden 8 \
  --initial_temp 1.0 \
  --final_temp 0.2 \
  --anneal_epochs 10 \
  --learning_rate 0.0003 \
  --des 'Exp2' \
  --itr 1


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path FULL.feather \
  --model_id FULL_run \
  --model $model_name \
  --data FULL \
  --dim_input 437 \
  --max_depth 5 \
  --hidden_dim_expert 16 \
  --alpha_fs 1.5612797736173043 \
  --beta_fs 0.27362172081515695 \
  --use_gating_mlp 1 \
  --gating_mlp_hidden 8 \
  --initial_temp 1.0 \
  --final_temp 0.2 \
  --anneal_epochs 10 \
  --learning_rate 0.0003 \
  --des 'Exp3' \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path FULL.feather \
  --model_id FULL_run \
  --model $model_name \
  --data FULL \
  --dim_input 437 \
  --max_depth 5 \
  --hidden_dim_expert 16 \
  --alpha_fs 1.655283870859086 \
  --beta_fs 0.136322774223865 \
  --use_gating_mlp 1 \
  --gating_mlp_hidden 8 \
  --initial_temp 1.0 \
  --final_temp 0.2 \
  --anneal_epochs 10 \
  --learning_rate 0.0003 \
  --des 'Exp1' \
  --itr 1


