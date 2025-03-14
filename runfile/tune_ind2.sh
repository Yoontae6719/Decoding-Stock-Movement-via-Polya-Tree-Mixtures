export CUDA_VISIBLE_DEVICES=0
model_name=treeMoE


python -u run.py \
  --is_training 2 \
  --root_path ./dataset/ \
  --data_path MSFT.feather \
  --model_id MSFT_tune \
  --model $model_name \
  --data MSFT \
  --des 'LOSS100' \
  --optuna_metric loss \
  --n_trial 100 \
  --itr 1



python -u run.py \
  --is_training 2 \
  --root_path ./dataset/ \
  --data_path NVDA.feather \
  --model_id NVDA_tune \
  --model $model_name \
  --data NVDA \
  --des 'LOSS100' \
  --optuna_metric loss \
  --n_trial 100 \
  --itr 1


  



python -u run.py \
  --is_training 2 \
  --root_path ./dataset/ \
  --data_path MSFT.feather \
  --model_id MSFT_tune \
  --model $model_name \
  --data MSFT \
  --des 'LOSS200' \
  --optuna_metric loss \
  --n_trial 200 \
  --itr 1



python -u run.py \
  --is_training 2 \
  --root_path ./dataset/ \
  --data_path NVDA.feather \
  --model_id NVDA_tune \
  --model $model_name \
  --data NVDA \
  --des 'LOSS200' \
  --optuna_metric loss \
  --n_trial 200 \
  --itr 1


  



















