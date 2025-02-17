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
  --alpha_fs 0.7197373125603016 \
  --beta_fs 1.0292850454597233 \
  --use_gating_mlp 0 \
  --gating_mlp_hidden 16 \
  --initial_temp 1.0 \
  --final_temp 0.2 \
  --anneal_epochs 10 \
  --learning_rate 0.001 \
  --des 'Exp' \
  --itr 1
