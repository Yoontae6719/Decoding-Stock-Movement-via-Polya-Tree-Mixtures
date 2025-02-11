export CUDA_VISIBLE_DEVICES=0

model_name=treeMoE

python -u run.py \
  --is_training 2 \
  --root_path ./dataset/ \
  --data_path SNP.csv \
  --model_id SNP_Tune \
  --model $model_name \
  --data SNP \
  --des 'tuneLoss' \
  --optuna_metric loss \
  --n_trial 5 \
  --itr 1
