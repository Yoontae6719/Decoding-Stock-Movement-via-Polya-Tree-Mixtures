import os
import torch
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import itertools
import argparse
import json
from datetime import datetime
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef
from collections import defaultdict
from sklearn.model_selection import train_test_split

import xgboost as xgb
from catboost import CatBoost, Pool

from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler

from utils.inference_and_explain import (
    compute_global_importance_soft, 
    compute_global_importance_hard,
    compute_global_influence_soft,
    compute_global_influence_hard,
    gather_leaf_feature_selection_data,
    inference_and_explain,
    collect_leaf_influence_gamma_weight
)
from utils.result_to_df import (
    leaf_selection_to_dataframe,
    leaf_influence_to_dataframe,
    global_importance_to_dataframes,
    node_level_stats_to_dataframe
)


from models.treeMoE import Model

from xai_utils_ver_a.trading_strategy import (
    implement_trading_strategy,
    evaluate_multi_model_strategies,
    calculate_returns,
    calculate_performance_metrics
)



# Step 0. Setting up a reproducible experimental environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fix_seed = 2025
random.seed(fix_seed)
np.random.seed(fix_seed)
torch.manual_seed(fix_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(fix_seed)



# Step 1. A function that converts NumPy types to JSON serializable Python native types
def convert_numpy_types(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [convert_numpy_types(i) for i in obj]
    else:
        return obj






# Step 3. Get load model & data loader
def load_model(args_model, checkpoint_path):
    model = Model(args_model).to(device)
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[INFO] Model loaded from {checkpoint_path}")
    else:
        print(f"[WARNING] checkpoint not found at {checkpoint_path}; using untrained model.")
    return model


# Step 6. Create a mask with only selected variables set to 1 and the rest set to 0
def create_feature_mask(selected_features, all_features):
    mask = np.zeros(len(all_features))
    for feature in selected_features:
        idx = all_features.index(feature)
        mask[idx] = 1.0
    return mask


# Step 7. Vis
def compare_strategies_performance(strategies_results, save_path=None):
    performance_data = {}
    for strat in strategies_results:
        name = strat['name']
        perf = strat['performance']
        performance_data[name] = perf
    
    df_performance = pd.DataFrame(performance_data).T
    
    plt.figure(figsize=(12, 10))
    
    # 1. Compare cumulative returns
    plt.subplot(2, 2, 1)
    for strat in strategies_results:
        plt.plot(strat['cumulative_returns'], label=f"{strat['name']} (n={strat['num_features']})")
    plt.title('Cumulative Returns Comparison')
    plt.legend()
    plt.grid(True)
    
    # 2. Comparison of Key Performance Indicators (Bar Graph)
    plt.subplot(2, 2, 2)
    metrics = ['total_return', 'sharpe_ratio', 'win_rate']
    for i, metric in enumerate(metrics):
        values = [strat['performance'][metric] for strat in strategies_results]
        x = np.arange(len(strategies_results))
        width = 0.25
        plt.bar(x + i*width, values, width=width, label=metric)
    
    plt.xticks(x + width, [strat['name'] for strat in strategies_results], rotation=45)
    plt.title('Key Performance Metrics')
    plt.legend()
    plt.grid(True)
    
    # 3. Monthly returns heatmap
    plt.subplot(2, 2, 3)
    strategy_returns = [strat['returns'] for strat in strategies_results]
    dates = pd.to_datetime(strategies_results[0].get('dates', pd.date_range(end='2024-02-01', periods=len(strategy_returns[0]), freq='D')))
    
    # Monthly returns
    monthly_returns = {}
    for i, strat in enumerate(strategies_results):
        returns = pd.Series(strat['returns'], index=dates)
        monthly = returns.resample('M').sum()
        monthly_returns[strat['name']] = monthly
    
    df_monthly = pd.DataFrame(monthly_returns)
    
    # Heat map
    sns.heatmap(df_monthly, cmap='RdYlGn', center=0, annot=True, fmt='.2%')
    plt.title('Monthly Returns Heatmap')
    
    # 4. win_rate AND profit and loss ratio
    plt.subplot(2, 2, 4)
    win_rates = [strat['performance']['win_rate'] for strat in strategies_results]
    pl_ratios = [strat['performance']['profit_loss_ratio'] for strat in strategies_results]
    
    x = np.arange(len(strategies_results))
    width = 0.35
    
    plt.bar(x - width/2, win_rates, width=width, label='Win Rate')
    plt.bar(x + width/2, pl_ratios, width=width, label='P/L Ratio')
    
    plt.xticks(x, [strat['name'] for strat in strategies_results], rotation=45)
    plt.title('Win Rate and P/L Ratio')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Performance comparison saved to {save_path}")
    
    plt.show()
    
    return df_performance



def analyze_feature_interactions(model, data_dict, top_features, gating_mode='hard', temperature=0.1, k_pairs=10):
    """Analysis of interactions between key variables"""
    feature_names = data_dict['feature_names']
    X = torch.tensor(data_dict['X'], dtype=torch.float32).to(device)
    close_prices = data_dict['close_prices']
    
    # Baseline (USE ALL FEATURES)
    full_results = implement_trading_strategy(model, data_dict, None, True, gating_mode, temperature, "All_Features")
    baseline_perf = full_results['performance']['total_return']
    
    print("[INFO] Calculating individual feature performance...")
    single_perfs = {}
    for feat in tqdm(top_features):
        results = implement_trading_strategy(
            model, data_dict, [feat], False, gating_mode, temperature, f"Single_{feat}"
        )
        single_perfs[feat] = results['performance']['total_return']
    
    print("[INFO] Calculating pairwise feature interactions...")
    pair_perfs = {}
    synergy_scores = {}
    
    feat_pairs = list(itertools.combinations(top_features, 2))
    for pair in tqdm(feat_pairs):
        feat1, feat2 = pair
        pair_name = f"{feat1}_{feat2}"
        
        results = implement_trading_strategy(
            model, data_dict, list(pair), False, gating_mode, temperature, f"Pair_{pair_name}"
        )
        pair_perf = results['performance']['total_return']
        pair_perfs[pair_name] = pair_perf
        
        # Calculate Synergy Score (Sum of Pair Performance - Individual Performance)
        individual_sum = single_perfs[feat1] + single_perfs[feat2]
        synergy = pair_perf - individual_sum
        synergy_scores[pair_name] = synergy
    
    # Selecting the pair with the highest synergy
    top_synergy_pairs = sorted(synergy_scores.items(), key=lambda x: x[1], reverse=True)[:k_pairs]
    
    plt.figure(figsize=(14, 8))
    
    plt.subplot(1, 2, 1)
    synergy_values = [score for _, score in top_synergy_pairs]
    synergy_names = [name for name, _ in top_synergy_pairs]
    
    plt.barh(range(len(synergy_names)), synergy_values, color='skyblue')
    plt.yticks(range(len(synergy_names)), synergy_names)
    plt.title('Top Feature Pairs by Synergy Score')
    plt.xlabel('Synergy Score (Pair Performance - Sum of Individual Performances)')
    plt.grid(True, axis='x')
    
    plt.subplot(1, 2, 2)    
    plt.tight_layout()
    plt.show()
    
    return {
        'baseline_performance': baseline_perf,
        'single_performances': single_perfs,
        'pair_performances': pair_perfs,
        'synergy_scores': synergy_scores,
        'top_synergy_pairs': top_synergy_pairs
    }



def evaluate_feature_combinations(model, data_dict, top_features, gating_mode='hard', temperature=0.1):
    """Evaluate various combinations of key variables"""
    feature_names = data_dict['feature_names']
    
    # Baseline results (ALL_results)
    baseline_results = implement_trading_strategy(
        model, data_dict, None, True, gating_mode, temperature, "All_Features"
    )
    
    # 점진적으로 변수 추가
    incremental_results = []
    for i in range(1, len(top_features) + 1):
        features_subset = top_features[:i]
        results = implement_trading_strategy(
            model, data_dict, features_subset, False, gating_mode, temperature, 
            f"Top_{i}_Features"
        )
        incremental_results.append({
            'num_features': i,
            'features': features_subset,
            'performance': results['performance']
        })
    
    # VIS
    plt.figure(figsize=(10, 6))
    
    x = [r['num_features'] for r in incremental_results]
    y_returns = [r['performance']['total_return'] for r in incremental_results]
    y_sharpe = [r['performance']['sharpe_ratio'] for r in incremental_results]
    
    plt.plot(x, y_returns, 'o-', label='Total Return')
    plt.plot(x, y_sharpe, 's-', label='Sharpe Ratio')
    plt.axhline(y=baseline_results['performance']['total_return'], linestyle='--', color='r', 
                label='Baseline Return')
    plt.axhline(y=baseline_results['performance']['sharpe_ratio'], linestyle='--', color='g', 
                label='Baseline Sharpe')
    
    plt.xlabel('Number of Top Features Used')
    plt.ylabel('Performance Metric')
    plt.title('Performance by Incrementally Adding Top Features')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return {
        'baseline': baseline_results['performance'],
        'incremental': incremental_results
    }


# Step 8 Market condition. Note that The analysis is meaningful if the market conditions exist sufficiently within the validation, although the market conditions may actually change in the future.
    
def classify_market_conditions(returns, prices, window=20):
    """Define Market Conditions"""
    
    ma_short = np.convolve(prices, np.ones(window)/window, mode='valid') # Calculate moving average
    ma_short = np.concatenate([np.full(window-1, ma_short[0]), ma_short]) # Add paddings
    
    volatility = np.zeros_like(prices) # Calculate Volatility using rolling std
    for i in range(window, len(prices)):
        volatility[i] = np.std(returns[i-window:i])
    volatility[:window] = volatility[window] # Paddings

    market_conditions = np.full(len(prices), 'unknown', dtype=object) # Market conditions
    
    for i in range(window, len(prices)):
        # Check Trend 
        if prices[i] > ma_short[i] * 1.03:
            trend = 'uptrend'
        elif prices[i] < ma_short[i] * 0.97:
            trend = 'downtrend'
        else:
            trend = 'sideways'
        
        # Check Volatitliy
        vol_median = np.median(volatility[window:])
        if volatility[i] > vol_median * 1.5:
            vol = 'high_vol'
        elif volatility[i] < vol_median * 0.5:
            vol = 'low_vol'
        else:
            vol = 'normal_vol'
        
        market_conditions[i] = f"{trend}_{vol}"
    
    return market_conditions

def evaluate_by_market_condition(strategies_results, market_conditions):
    """Performance evaluation of strategies according to market conditions"""
    unique_conditions = np.unique(market_conditions)
    condition_performance = {}
    
    for condition in unique_conditions:
        if condition == 'unknown':
            continue
            
        mask = (market_conditions == condition) # Date mask corresponding to the market situation
        if np.sum(mask) < 5:  # Skip if there are too few data points
            continue
        
        condition_performance[condition] = {}
        
        for strat in strategies_results:
            returns_in_condition = strat['returns'][mask] # Returns in the market
            
            if len(returns_in_condition) == 0:
                continue
                
            # Calcualte perfomance indication
            avg_return = np.mean(returns_in_condition)
            win_rate = np.sum(returns_in_condition > 0) / len(returns_in_condition)
            
            condition_performance[condition][strat['name']] = {
                'avg_return': avg_return,
                'win_rate': win_rate,
                'count': np.sum(mask)
            }
    
    # vis
    plt.figure(figsize=(15, 10))
    
    plot_data = {}
    for condition in condition_performance:
        for strat_name in condition_performance[condition]:
            if strat_name not in plot_data:
                plot_data[strat_name] = {'conditions': [], 'returns': []}
            plot_data[strat_name]['conditions'].append(condition)
            plot_data[strat_name]['returns'].append(condition_performance[condition][strat_name]['avg_return'])
    
    bar_width = 0.8 / len(plot_data)
    
    for i, (strat_name, data) in enumerate(plot_data.items()):
        x = np.arange(len(data['conditions']))
        offset = (i - len(plot_data)/2 + 0.5) * bar_width
        plt.bar(x + offset, data['returns'], bar_width, label=strat_name)
    
    plt.xticks(np.arange(len(list(condition_performance.keys()))), list(condition_performance.keys()), rotation=45)
    plt.title('Average Returns by Market Condition')
    plt.xlabel('Market Condition')
    plt.ylabel('Average Daily Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    heatmap_data = []
    for condition in condition_performance:
        for strat_name in condition_performance[condition]:
            heatmap_data.append({
                'condition': condition,
                'strategy': strat_name,
                'return': condition_performance[condition][strat_name]['avg_return'],
                'win_rate': condition_performance[condition][strat_name]['win_rate'],
                'count': condition_performance[condition][strat_name]['count']
            })
    
    df_heatmap = pd.DataFrame(heatmap_data)
    df_pivot = df_heatmap.pivot(index='strategy', columns='condition', values='return')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_pivot, cmap='RdYlGn', center=0, annot=True, fmt='.2%')
    plt.title('Strategy Performance Heatmap by Market Condition (Daily Return)')
    plt.tight_layout()
    plt.show()
    
    return condition_performance


# Step 9. Statstisitcal test

def statistical_significance_test(strategies_results):
    """Statistical significance test of performance differences between strategies"""
    strategy_names = [strat['name'] for strat in strategies_results]
    
    returns_dict = {strat['name']: strat['returns'] for strat in strategies_results}
    
    n_bootstrap = 1000
    bootstrap_results = {}
    
    for name, returns in returns_dict.items():
        bootstrap_samples = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(returns), len(returns), replace=True)
            sample = returns[indices]
            total_return = np.prod(1 + sample) - 1
            bootstrap_samples.append(total_return)
        
        # 95% CI
        ci_lower = np.percentile(bootstrap_samples, 2.5)
        ci_upper = np.percentile(bootstrap_samples, 97.5)
        
        bootstrap_results[name] = {
            'mean': np.mean(bootstrap_samples),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'samples': bootstrap_samples
        }
    
    t_test_results = {}
    for i in range(len(strategy_names)):
        for j in range(i+1, len(strategy_names)):
            name_i, name_j = strategy_names[i], strategy_names[j]
            
            t_stat, p_value = stats.ttest_ind(
                bootstrap_results[name_i]['samples'],
                bootstrap_results[name_j]['samples'],
                equal_var=False  # Welch's t-test
            )
            
            t_test_results[f"{name_i}_vs_{name_j}"] = {
                't_stat': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
    
    plt.figure(figsize=(10, 6))
    
    # CI using Bootstrap[
    for i, name in enumerate(strategy_names):
        mean = bootstrap_results[name]['mean']
        ci_lower = bootstrap_results[name]['ci_lower']
        ci_upper = bootstrap_results[name]['ci_upper']
        
        plt.errorbar(i, mean, yerr=[[mean - ci_lower], [ci_upper - mean]], 
                     fmt='o', capsize=5, label=name)
    
    plt.xticks(range(len(strategy_names)), strategy_names, rotation=45)
    plt.title('Bootstrap Mean Total Return with 95% Confidence Intervals')
    plt.xlabel('Strategy')
    plt.ylabel('Total Return')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # t-test
    fig, ax = plt.subplots(figsize=(12, len(t_test_results) * 0.5))
    
    comparison_names = list(t_test_results.keys())
    p_values = [t_test_results[name]['p_value'] for name in comparison_names]
    significant = [t_test_results[name]['significant'] for name in comparison_names]
    
    colors = ['green' if sig else 'red' for sig in significant]
    
    ax.barh(comparison_names, p_values, color=colors)
    ax.axvline(x=0.05, linestyle='--', color='black', label='p=0.05')
    ax.set_title('P-values for Strategy Comparisons')
    ax.set_xlabel('P-value')
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    
    return {
        'bootstrap': bootstrap_results,
        't_tests': t_test_results
    }


# Step 10. XG and Cat boost (Default Settings)
    
def train_tree_models(data_loader, output_dir='./models'):
    os.makedirs(output_dir, exist_ok=True)
    
    print("[INFO] Preparing data for tree models...")
    all_x, all_y = [], []
    for batch_x, batch_y, _, _ in data_loader:
        all_x.append(batch_x.numpy())
        all_y.append(batch_y.numpy())
    
    X = np.vstack(all_x)
    y = np.vstack(all_y).ravel()
    
    feature_names = data_loader.dataset.feature_names
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=fix_seed)
    
    print("[INFO] Training XGBoost model with default parameters...")
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=fix_seed)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    xgb_importance = xgb_model.feature_importances_
    xgb_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb_importance
    }).sort_values('importance', ascending=False)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    xgb_importance_df.to_csv(f"{output_dir}/xgboost_importance_{timestamp}.csv", index=False)
    
    xgb_model.save_model(f"{output_dir}/xgboost_model_{timestamp}.json")
    
    print("[INFO] Training CatBoost model with default parameters...")
    cat_model = CatBoost(params={'loss_function': 'Logloss', 'verbose': False, 'random_seed': fix_seed})
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    
    cat_importance = cat_model.get_feature_importance()
    cat_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': cat_importance
    }).sort_values('importance', ascending=False)
    
    cat_importance_df.to_csv(f"{output_dir}/catboost_importance_{timestamp}.csv", index=False)
    
    cat_model.save_model(f"{output_dir}/catboost_model_{timestamp}")
    
    print(f"[INFO] Tree models trained and saved to {output_dir}")
    
    return {
        'xgboost': {
            'model': xgb_model,
            'importance': xgb_importance,
            'importance_df': xgb_importance_df
        },
        'catboost': {
            'model': cat_model,
            'importance': cat_importance,
            'importance_df': cat_importance_df
        },
        'feature_names': feature_names
    }


















































































































