export CUDA_VISIBLE_DEVICES=0

model_name=treeMoE

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path SNP.csv \
  --model_id SNP_Test \
  --model $model_name \
  --data SNP \
  --dim_input 437 \
  --max_depth 5 \
  --hidden_dim_expert 32 \
  --alpha_fs 1.0 \
  --beta_fs 1.0 \
  --use_gating_mlp 0 \
  --des 'Exp' \
  --itr 1
