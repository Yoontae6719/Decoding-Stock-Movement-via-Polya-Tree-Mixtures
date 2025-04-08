#!/bin/bash
# Script for running enhanced trading analysis with the new metrics

# Create necessary directories
mkdir -p trading
mkdir -p trading/analysis
mkdir -p trading/analysis/summary

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Default parameters
TOP_K=10

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --top_k)
      TOP_K="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

echo "Using TOP_K = $TOP_K features"

# Function to run trading analysis for a stock
run_trading_analysis() {
    local stock_symbol=$1
    local max_depth=$2
    local hidden_dim_expert=$3
    local alpha_fs=$4
    local use_leaf_fs=$5
    local use_gating_mlp=$6
    local gating_mlp_hidden=$7
    local top_k=${8:-$TOP_K}
    
    echo "Running analysis for ${stock_symbol}..."
    
    python run_trading.py \
        --model_id ${stock_symbol} \
        --data ${stock_symbol} \
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
        --des trading \
        --initial_temp 1.0 \
        --final_temp 0.1 \
        --anneal_epochs 5 \
        --schedule_type linear \
        --train_epochs 100 \
        --learning_rate 0.0003 \
        --lambda_KL 0.01 \
        --batch_size 16 \
        --patience 5 \
        --top_k ${top_k}
}

# Function to run trading analysis with existing model (no training)
run_trading_analysis_existing() {
    local stock_symbol=$1
    local max_depth=$2
    local hidden_dim_expert=$3
    local alpha_fs=$4
    local use_leaf_fs=$5
    local use_gating_mlp=$6
    local gating_mlp_hidden=$7
    local top_k=${8:-$TOP_K}
    
    echo "Running analysis with existing model for ${stock_symbol}..."
    
    python run_trading.py \
        --model_id ${stock_symbol} \
        --data ${stock_symbol} \
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
        --train_epochs 0 \
        --learning_rate 0.0003 \
        --lambda_KL 0.01 \
        --batch_size 16 \
        --patience 5 \
        --top_k ${top_k} \
        --use_existing_model
}

# Choose run mode
# Options: "parallel" (run all in parallel), "sequential" (run one by one), 
# "existing" (use existing models), "consolidate" (only consolidate results)
MODE="existing"

if [ "$MODE" = "consolidate" ]; then
    # Only consolidate results without running analysis
    python run_trading.py --model_id NONE --data NONE --consolidate_only
    echo "Consolidation completed!"
    exit 0
fi

run_stocks() {
    # Run all stocks based on the selected mode
    if [ "$MODE" = "sequential" ]; then
        # Run sequentially
        run_trading_analysis "AAPL" 3 32 0.5 1 1 8 $TOP_K
        run_trading_analysis "TSLA" 2 32 0.9 0 0 32 $TOP_K
        run_trading_analysis "META" 2 16 0.9 0 0 32 $TOP_K
        run_trading_analysis "NFLX" 3 16 0.7 1 1 8 $TOP_K
        run_trading_analysis "NVDA" 4 8 0.7 0 1 16 $TOP_K
        run_trading_analysis "AMZN" 4 32 0.9 0 1 16 $TOP_K
        run_trading_analysis "PEP" 2 16 0.9 1 1 8 $TOP_K
        run_trading_analysis "JPM" 2 32 0.9 0 1 16 $TOP_K
        run_trading_analysis "BNTX" 2 16 0.9 1 1 16 $TOP_K
    
    elif [ "$MODE" = "parallel" ]; then
        # Run in parallel
        run_trading_analysis "AAPL" 3 32 0.5 1 1 8 $TOP_K &
        run_trading_analysis "TSLA" 2 32 0.9 0 0 32 $TOP_K &
        run_trading_analysis "META" 2 16 0.9 0 0 32 $TOP_K &
        run_trading_analysis "NFLX" 3 16 0.7 1 1 8 $TOP_K &
        run_trading_analysis "NVDA" 4 8 0.7 0 1 16 $TOP_K &
        run_trading_analysis "AMZN" 4 32 0.9 0 1 16 $TOP_K &
        run_trading_analysis "PEP" 2 16 0.9 1 1 8 $TOP_K &
        run_trading_analysis "JPM" 2 32 0.9 0 1 16 $TOP_K &
        run_trading_analysis "BNTX" 2 16 0.9 1 1 16 $TOP_K &
        
        # Wait for all background processes to finish
        wait
    
    elif [ "$MODE" = "existing" ]; then
        # Use existing models
        run_trading_analysis_existing "AAPL" 3 32 0.5 1 1 8 $TOP_K &
        run_trading_analysis_existing "TSLA" 2 32 0.9 0 0 32 $TOP_K &
        run_trading_analysis_existing "META" 2 16 0.9 0 0 32 $TOP_K &
        run_trading_analysis_existing "NFLX" 3 16 0.7 1 1 8 $TOP_K &
        run_trading_analysis_existing "NVDA" 4 8 0.7 0 1 16 $TOP_K &
        run_trading_analysis_existing "AMZN" 4 32 0.9 0 1 16 $TOP_K &
        run_trading_analysis_existing "PEP" 2 16 0.9 1 1 8 $TOP_K &
        run_trading_analysis_existing "JPM" 2 32 0.9 0 1 16 $TOP_K &
        run_trading_analysis_existing "BNTX" 2 16 0.9 1 1 16 $TOP_K &
        
        # Wait for all background processes to finish
        wait
    fi
}

# Run the stocks
run_stocks

# Consolidate results at the end
python run_trading.py --model_id NONE --data NONE --consolidate_only

echo "Trading analysis and consolidation completed!"