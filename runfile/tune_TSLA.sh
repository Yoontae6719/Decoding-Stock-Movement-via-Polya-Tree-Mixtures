export CUDA_VISIBLE_DEVICES=0

model_name=treeMoE


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path TSLA.feather \
  --model_id TSLA_run \
  --model $model_name \
  --data TSLA \
  --dim_input 437 \
  --alpha_fs 0.3302208363198128 \
  --beta_fs 0.1907433014660887 \
  --max_depth 1 \
  --use_gating_mlp 1 \
  --gating_mlp_hidden 32 \
  --hidden_dim_expert 32 \
  --anneal_epochs 30 \
  --schedule_type linear \
  --learning_rate 0.0003 \
  --lradj type2 \
  --des 'EXPNEW' \
  --itr 1



