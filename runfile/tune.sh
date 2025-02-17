export CUDA_VISIBLE_DEVICES=0

model_name=treeMoE

python -u run.py \
  --is_training 2 \
  --root_path ./dataset/ \
  --data_path FULL.feather \
  --model_id FULL_tune \
  --model $model_name \
  --data FULL \
  --des 'tuneLoss' \
  --optuna_metric loss \
  --n_trial 150 \
  --itr 1
