export CUDA_VISIBLE_DEVICES=0

model_name=treeMoE


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path AAPL.feather \
  --model_id AAPL_run \
  --model $model_name \
  --data AAPL \
  --dim_input 437 \
  --alpha_fs 0.10060511161375214 \
  --beta_fs 0.8725602600261494 \
  --max_depth 2 \
  --use_gating_mlp 0 \
  --gating_mlp_hidden 8 \
  --hidden_dim_expert 32 \
  --initial_temp 2.0 \
  --final_temp 0.2 \
  --anneal_epochs 10 \
  --schedule_type linear \
  --learning_rate 0.003 \
  --lradj cosine \
  --des 'EXPNEW' \
  --itr 2



