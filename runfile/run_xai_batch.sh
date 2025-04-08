#!/bin/bash
# Script for running XAI analysis integrated with model training

# Create runfile directory if it doesn't exist
mkdir -p runfile

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Function to run training with integrated XAI analysis
run_train_with_xai() {
    local stock_symbol=$1
    local max_depth=$2
    local hidden_dim_expert=$3
    local alpha_fs=$4
    local use_leaf_fs=$5
    local use_gating_mlp=$6
    local gating_mlp_hidden=$7
    
    echo "Running training with integrated XAI for ${stock_symbol}..."
    
    python run.py \
        --is_training 1 \
        --model treeMoE \
        --data ${stock_symbol} \
        --model_id ${stock_symbol} \
        --root_path ./dataset/ \
        --data_path ${stock_symbol}.feather \
        --dim_input 437 \
        --num_classes 2 \
        --max_depth ${max_depth} \
        --hidden_dim_expert ${hidden_dim_expert} \
        --alpha_fs ${alpha_fs} \
        --beta_fs 1.0 \
        --use_feature_selection \
        --use_leaf_feature_selector_only ${use_leaf_fs} \
        --use_gating_mlp ${use_gating_mlp} \
        --gating_mlp_hidden ${gating_mlp_hidden} \
        --lradj cosine \
        --des EXPNEW \
        --initial_temp 1.0 \
        --final_temp 0.1 \
        --anneal_epochs 5 \
        --schedule_type linear \
        --train_epochs 100 \
        --learning_rate 0.0003 \
        --lambda_KL 0.01 \
        --batch_size 16 \
        --patience 5 \
        --itr 1
}

# Function to run standalone XAI analysis on already trained models
run_standalone_xai() {
    local stock_symbol=$1
    local max_depth=$2
    local hidden_dim_expert=$3
    local alpha_fs=$4
    local use_leaf_fs=$5
    local use_gating_mlp=$6
    local gating_mlp_hidden=$7
    
    echo "Running standalone XAI analysis for ${stock_symbol}..."
    
    python run.py \
        --is_training 0 \
        --model treeMoE \
        --data ${stock_symbol} \
        --model_id ${stock_symbol} \
        --root_path ./dataset/ \
        --data_path ${stock_symbol}.feather \
        --dim_input 437 \
        --num_classes 2 \
        --max_depth ${max_depth} \
        --hidden_dim_expert ${hidden_dim_expert} \
        --alpha_fs ${alpha_fs} \
        --beta_fs 1.0 \
        --use_feature_selection \
        --use_leaf_feature_selector_only ${use_leaf_fs} \
        --use_gating_mlp ${use_gating_mlp} \
        --gating_mlp_hidden ${gating_mlp_hidden} \
        --lradj cosine \
        --des EXPNEW \
        --final_temp 0.1
}

# Mode selection (choose one)
mode="train_with_xai"  # Options: "train_with_xai" or "standalone_xai"

# Process stocks based on mode
if [ "$mode" = "train_with_xai" ]; then
    # Train models with integrated XAI
    run_train_with_xai "AAPL" 3 32 0.5 1 1 8
    run_train_with_xai "TSLA" 2 32 0.9 0 0 32
    run_train_with_xai "PEP" 2 16 0.9 1 1 8
    run_train_with_xai "JPM" 2 32 0.9 1 1 16
    run_train_with_xai "AMZN" 4 32 0.9 0 0 16
    run_train_with_xai "META" 2 16 0.9 0 0 32
    run_train_with_xai "NVDA" 4 8 0.7 0 1 16
    run_train_with_xai "NFLX" 3 16 0.7 1 1 8
    run_train_with_xai "BNTX" 2 16 0.9 1 1 16

else
    run_train_with_xai "AAPL" 3 32 0.5 1 1 8
    run_train_with_xai "TSLA" 2 32 0.9 0 0 32
    run_train_with_xai "PEP" 2 16 0.9 1 1 8
    run_train_with_xai "JPM" 2 32 0.9 1 1 16
    run_train_with_xai "AMZN" 4 32 0.9 0 0 16
    run_train_with_xai "META" 2 16 0.9 0 0 32
    run_train_with_xai "NVDA" 4 8 0.7 0 1 16
    run_train_with_xai "NFLX" 3 16 0.7 1 1 8
    run_train_with_xai "BNTX" 2 16 0.9 1 1 16
fi