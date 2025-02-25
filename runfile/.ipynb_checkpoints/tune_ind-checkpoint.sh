export CUDA_VISIBLE_DEVICES=0
model_name=treeMoE

python -u run.py \
  --is_training 2 \
  --root_path ./dataset/ \
  --data_path TSLA.feather \
  --model_id TSLA_tune \
  --model $model_name \
  --data TSLA \
  --des 'LOSS100' \
  --optuna_metric loss \
  --n_trial 100 \
  --itr 1

python -u run.py \
  --is_training 2 \
  --root_path ./dataset/ \
  --data_path AAPL.feather \
  --model_id AAPL_tune \
  --model $model_name \
  --data AAPL \
  --des 'LOSS100' \
  --optuna_metric loss \
  --n_trial 100 \
  --itr 1

python -u run.py \
  --is_training 2 \
  --root_path ./dataset/ \
  --data_path AMZN.feather \
  --model_id AMZN_tune \
  --model $model_name \
  --data AMZN \
  --des 'LOSS100' \
  --optuna_metric loss \
  --n_trial 100 \
  --itr 1


python -u run.py \
  --is_training 2 \
  --root_path ./dataset/ \
  --data_path JPM.feather \
  --model_id JPM_tune \
  --model $model_name \
  --data JPM \
  --des 'LOSS100' \
  --optuna_metric loss \
  --n_trial 100 \
  --itr 1


python -u run.py \
  --is_training 2 \
  --root_path ./dataset/ \
  --data_path META.feather \
  --model_id META_tune \
  --model $model_name \
  --data META \
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
  --data MSFT \
  --des 'LOSS100' \
  --optuna_metric loss \
  --n_trial 100 \
  --itr 1



















