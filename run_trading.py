#!/usr/bin/env python
# run_trading.py

import argparse
import os
import torch
import torch.backends
import random
import numpy as np
from exp.exp_trading import Exp_Trading

def run_trading_analysis(args):
    print(f"Running trading analysis for {args.model_id}...")
    print(f"Using top_k={args.top_k} features with local importance weighting")
    
    os.makedirs("./trading", exist_ok=True)
    os.makedirs(f"./trading/trading_{args.data}", exist_ok=True)
    os.makedirs("./trading/analysis", exist_ok=True)
    os.makedirs(f"./trading/analysis/{args.data}", exist_ok=True)
    
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device(f'cuda:{args.gpu}')
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using CPU or MPS')
    
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    setting = f"{args.model}_{args.data}_md{args.max_depth}_hde{args.hidden_dim_expert}_alp{int(args.alpha_fs * 10)}_use{args.use_leaf_feature_selector_only}_mlp{args.use_gating_mlp}_gde_{args.gating_mlp_hidden}_lr_{args.lradj}_des{args.des}_0"
    
    exp = Exp_Trading(args)
    
    checkpoint_path = os.path.join(args.checkpoints, setting, 'checkpoint.pth')
    if os.path.exists(checkpoint_path) and args.use_existing_model:
        print(f"Loading existing model from {checkpoint_path}")
        exp.model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))
        print("Model loaded successfully")
    
    results = exp.trading(setting)
    
    print(f"Trading analysis for {args.model_id} completed!")
    
    return results

def consolidate_results():
    import pandas as pd
    import glob
    
    print("Consolidating results from all stocks...")
    
    os.makedirs("./trading/analysis/summary", exist_ok=True)    
    analysis_files = glob.glob("./trading/analysis/*/[A-Z]*_analysis_results.csv")
    
    if not analysis_files:
        print("No analysis result files found.")
        return
    
    all_dfs = []
    for file in analysis_files:
        try:
            df = pd.read_csv(file)
            all_dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not all_dfs:
        print("No valid data to consolidate.")
        return
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    combined_df.to_csv("./trading/analysis/summary/all_analysis_results.csv", index=False)
    
    strategy_summary = combined_df.groupby(['Strategy']).agg({
        'Accuracy': ['mean', 'std', 'min', 'max'],
        'MCC': ['mean', 'std', 'min', 'max'],
        'Sharpe_Ratio': ['mean', 'std', 'min', 'max'],
        'Final_Return': ['mean', 'std', 'min', 'max']
    })
    
    strategy_summary.to_csv("./trading/analysis/summary/strategy_summary.csv")
    
    stock_summary = combined_df.groupby(['Stock']).agg({
        'Accuracy': ['mean', 'std'],
        'MCC': ['mean', 'std'],
        'Sharpe_Ratio': ['mean', 'std'],
        'Final_Return': ['mean', 'std']
    })
    
    stock_summary.to_csv("./trading/analysis/summary/stock_summary.csv")
    
    feature_type_summary = combined_df.groupby(['Feature_Type']).agg({
        'Accuracy': ['mean', 'std', 'min', 'max'],
        'MCC': ['mean', 'std', 'min', 'max'],
        'Sharpe_Ratio': ['mean', 'std', 'min', 'max'],
        'Final_Return': ['mean', 'std', 'min', 'max']
    })
    
    feature_type_summary.to_csv("./trading/analysis/summary/feature_type_summary.csv")
    
    importance_metrics = ['global_importance', 'feature_selection_frequency', 
                         'phi_importance', 'coverage_weighted_importance']
    
    for metric in importance_metrics:
        top_filter = combined_df['Strategy'].str.contains(f"{metric}_top")
        bottom_filter = combined_df['Strategy'].str.contains(f"{metric}_bottom")
        
        if top_filter.any() and bottom_filter.any():
            top_data = combined_df[top_filter]
            bottom_data = combined_df[bottom_filter]
            
            comparison = pd.DataFrame({
                'Top_Mean_Accuracy': top_data['Accuracy'].mean(),
                'Bottom_Mean_Accuracy': bottom_data['Accuracy'].mean(),
                'Top_Mean_MCC': top_data['MCC'].mean(),
                'Bottom_Mean_MCC': bottom_data['MCC'].mean(),
                'Top_Mean_Sharpe': top_data['Sharpe_Ratio'].mean(),
                'Bottom_Mean_Sharpe': bottom_data['Sharpe_Ratio'].mean(),
                'Top_Mean_Return': top_data['Final_Return'].mean(),
                'Bottom_Mean_Return': bottom_data['Final_Return'].mean(),
            }, index=[metric])
            
            comparison.to_csv(f"./trading/analysis/summary/{metric}_comparison.csv")
    
    try:
        import matplotlib.pyplot as plt        
        plt.figure(figsize=(16, 12))
        
        for i, metric in enumerate(importance_metrics):
            plt.subplot(2, 2, i+1)
            
            top_data = combined_df[combined_df['Strategy'].str.contains(f"{metric}_top")]
            bottom_data = combined_df[combined_df['Strategy'].str.contains(f"{metric}_bottom")]
            
            if not top_data.empty and not bottom_data.empty:
                top_means = [
                    top_data['Accuracy'].mean(),
                    top_data['MCC'].mean(),
                    top_data['Sharpe_Ratio'].mean(),
                    top_data['Final_Return'].mean()
                ]
                
                bottom_means = [
                    bottom_data['Accuracy'].mean(),
                    bottom_data['MCC'].mean(), 
                    bottom_data['Sharpe_Ratio'].mean(),
                    bottom_data['Final_Return'].mean()
                ]
                
                x = np.arange(4) 
                width = 0.35
                
                plt.bar(x - width/2, top_means, width, label='Top Features', color='green', alpha=0.7)
                plt.bar(x + width/2, bottom_means, width, label='Bottom Features', color='red', alpha=0.7)
                
                plt.ylabel('Average Value')
                plt.title(f'{metric} - Top vs Bottom Features')
                plt.xticks(x, ['Accuracy', 'MCC', 'Sharpe', 'Return'])
                plt.legend()
                
        plt.tight_layout()
        plt.savefig("./trading/analysis/summary/importance_comparison.png")
        plt.close()
        
        strategies = combined_df['Strategy'].unique()
        if len(strategies) > 1:
            metrics = ['Accuracy', 'MCC', 'Sharpe_Ratio', 'Final_Return']
            
            for metric in metrics:
                plt.figure(figsize=(14, 8))
                
                grouped = combined_df.groupby('Strategy')[metric]
                means = grouped.mean().reindex(strategies)
                stds = grouped.std().reindex(strategies)
                
                colors = ['green' if 'top' in s else 'red' for s in strategies]
                
                plt.bar(range(len(strategies)), means, yerr=stds, color=colors, alpha=0.7)
                plt.xticks(range(len(strategies)), strategies, rotation=45, ha='right')
                plt.ylabel(f'Average {metric}')
                plt.title(f'Comparison of Strategies by {metric}')
                plt.tight_layout()
                plt.savefig(f"./trading/analysis/summary/strategy_comparison_{metric}.png")
                plt.close()
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
    
    print("Results consolidation completed!")

if __name__ == "__main__":
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    
    parser = argparse.ArgumentParser(description='TreeMoE Trading Analysis')
    
    # Basic config
    parser.add_argument('--model_id', type=str, required=True, help='model id/stock symbol')
    parser.add_argument('--model', type=str, default='treeMoE', help='model name')
    parser.add_argument('--use_existing_model', action='store_true', help='use existing model instead of training')
    
    # Data loader
    parser.add_argument('--data', type=str, required=True, help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default=None, help='data file')
    parser.add_argument('--stop_loss', type=float, default=0, help='stop loss ratio')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--scale', type=bool, default=True, help='scale param')
    
    # Optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='trading', help='exp description')
    parser.add_argument('--lradj', type=str, default='cosine', help='adjust learning rate')
    parser.add_argument('--temperature_scheduler', type=bool, default=True, help='temperature scheduler')
    
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multiple gpus')
    
    # Models param
    parser.add_argument('--dim_input', type=int, default=437, help='input dimension')
    parser.add_argument('--num_classes', type=int, default=2, help='output dimension')
    parser.add_argument('--max_depth', type=int, default=2, help='tree max depth')
    parser.add_argument('--hidden_dim_expert', type=int, default=32, help='hidden dim for expert')
    parser.add_argument('--alpha_fs', type=float, default=0.9, help='Beta-Bernoulli alpha')
    parser.add_argument('--beta_fs', type=float, default=1.0, help='Beta-Bernoulli beta')
    parser.add_argument('--use_gating_mlp', type=int, default=1, help='use_gating_mlp')
    parser.add_argument('--gating_mlp_hidden', type=int, default=16, help='Gating mlp hidden dim')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='max_grad_norm')
    
    parser.add_argument('--initial_temp', type=float, default=1.0, help='initial_temp')
    parser.add_argument('--final_temp', type=float, default=0.1, help='final_temp')
    parser.add_argument('--anneal_epochs', type=int, default=5, help='anneal_epochs')
    parser.add_argument('--schedule_type', type=str, default="linear", help='schedule_type')
    
    parser.add_argument('--use_feature_selection', action='store_true', default=True, help='use feature selection')
    parser.add_argument('--use_leaf_feature_selector_only', type=int, default=1, help='use leaf feature selector only')
    parser.add_argument('--lambda_KL', type=float, default=0.01, help='lambda_KL')
    
    # Analysis parameters
    parser.add_argument('--top_k', type=int, default=5, help='number of top features to use')
    parser.add_argument('--use_local_importance', action='store_true', default=True, help='use local importance weights')
    parser.add_argument('--importance_weight_scale', type=float, default=1.0, help='scale factor for importance weights')
    parser.add_argument('--reliability_threshold', type=str, default='low', help='minimum reliability (low, medium, high)')
    parser.add_argument('--feature_sets', type=str, default='all', help='feature sets to analyze (all, top, bottom, mid, random)')
    parser.add_argument('--consolidate', action='store_true', help='consolidate results after running')
    parser.add_argument('--consolidate_only', action='store_true', help='only consolidate results without running analysis')
    
    args = parser.parse_args()
    
    if args.data_path is None:
        args.data_path = f"{args.model_id}.feather"
    
    if args.consolidate_only:
        consolidate_results()
    else:
        run_trading_analysis(args)
        if args.consolidate:
            consolidate_results()