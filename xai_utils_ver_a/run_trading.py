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


###################################################################################################################################################
# Utils
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



def select_features(importance_scores, feature_names, method='top_k', param=10):
    if method == 'top_k':
        indices = np.argsort(importance_scores)[-param:]
        return [feature_names[i] for i in indices]
    
    elif method == 'bottom_k':
        indices = np.argsort(importance_scores)[:param]
        return [feature_names[i] for i in indices]
    
    elif method == 'threshold':
        indices = np.where(importance_scores >= param)[0]
        return [feature_names[i] for i in indices]
    
    elif method == 'random':
        random.seed(fix_seed)
        indices = random.sample(range(len(feature_names)), param)
        return [feature_names[i] for i in indices]
    
    elif method == 'cumulative':
        sorted_idx = np.argsort(importance_scores)[::-1]
        sorted_importance = importance_scores[sorted_idx]
        cumsum = np.cumsum(sorted_importance) / np.sum(sorted_importance)
        n_features = np.where(cumsum >= param)[0][0] + 1
        return [feature_names[i] for i in sorted_idx[:n_features]]
    
    else:
        raise ValueError(f"Unknown method: {method}")



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


###################################################################################################################################################

# Step 1: TRADING STRATEGY
def implement_trading_strategy(model, data_dict, selected_features=None, use_all=False, gating_mode='hard', temperature=0.1, name=""):
    """Implement a trading strategy that utilizes only selected variables"""
    model.eval()
    
    X = torch.tensor(data_dict['X'], dtype=torch.float32).to(device)
    feature_names = data_dict['feature_names']
    close_prices = data_dict['close_prices']
    
    # Step 1. Create a selected variable mask
    if not use_all and selected_features is not None:
        feature_mask = create_feature_mask(selected_features, feature_names)
        feature_mask_tensor = torch.tensor(feature_mask, dtype=torch.float32).to(device)
        
        # Apply mask (keep selected variables only)
        X_masked = X * feature_mask_tensor
    else:
        X_masked = X
    
    # Predict
    with torch.no_grad():
        if gating_mode == 'hard':
            probs = model.forward_hard(X_masked)
        else:
            probs = model(X_masked, temperature=temperature)
    
    # Step 2. Generate buy/sell signals based on predictions
    probs_buy = probs[:, 1].cpu().numpy()  # BUY prob (class 1)
    
    # Step 3. Buying/Selling Decision (Basic Threshold 0.5)
    signals = np.where(probs_buy > 0.5, 1, 0)  # 1: buy, 0: sell
    
    # Steop4. Calculate returns
    returns, cumulative_returns = calculate_returns(signals, close_prices)
    
    #Steop5.  Calculate Performance indicator
    performance = calculate_performance_metrics(returns, cumulative_returns, signals, data_dict['Y'])
    
    return {
        'name': name,
        'selected_features': selected_features,
        'num_features': len(selected_features) if selected_features is not None else len(feature_names),
        'signals': signals,
        'returns': returns,
        'cumulative_returns': cumulative_returns,
        'performance': performance,
        'probabilities': probs_buy
    }



# Step 2. eval trading strategies
    
def evaluate_multi_model_strategies(moe_model, tree_models_data, data_dict, 
                                   gating_mode='hard', temperature=0.1,
                                   top_k=10, plot_top_k=10, plot_union_top_k = 10, output_dir='./XAI_results_a'):
    feature_names = data_dict['feature_names']
    
    # Extract important characteristics from each model
    moe_importance = data_dict['global_importance']['gamma_weight']['importance'] if 'gamma_weight' in data_dict['global_importance'] else data_dict['global_importance']['phi_only']['importance']
    xgb_importance = tree_models_data['xgboost']['importance']
    cat_importance = tree_models_data['catboost']['importance']
    
    # Select the upper/lower characteristics for each model
    moe_top = select_features(moe_importance, feature_names, 'top_k', top_k)
    moe_bottom = select_features(moe_importance, feature_names, 'bottom_k', top_k)
    
    xgb_top = select_features(xgb_importance, feature_names, 'top_k', top_k)
    xgb_bottom = select_features(xgb_importance, feature_names, 'bottom_k', top_k)
    
    cat_top = select_features(cat_importance, feature_names, 'top_k', top_k)
    cat_bottom = select_features(cat_importance, feature_names, 'bottom_k', top_k)
    
    # The intersection and union of the top characteristics of all models
    all_top_intersection = list(set(moe_top) & set(xgb_top) & set(cat_top))
    all_top_union = list(set().union(moe_top, xgb_top, cat_top))
    
    strategies_results = []
    
    # Use all Features
    all_features_results = implement_trading_strategy( moe_model, data_dict, None, True, gating_mode, temperature, "All_Features")
    strategies_results.append(all_features_results)
    
    # treeMoE top K
    moe_top_results = implement_trading_strategy(moe_model, data_dict, moe_top, False, gating_mode, temperature, "MoE_Top")
    strategies_results.append(moe_top_results)
    
    # XGBoost Top K
    xgb_top_results = implement_trading_strategy(moe_model, data_dict, xgb_top, False, gating_mode, temperature, "XGB_Top")
    strategies_results.append(xgb_top_results)
    
    # CatBoost  Top K
    cat_top_results = implement_trading_strategy(moe_model, data_dict, cat_top, False, gating_mode, temperature, "CAT_Top")
    strategies_results.append(cat_top_results)
    
    # Features(Techinical indicatores) of all models intersection
    if len(all_top_intersection) > 0:
        intersection_results = implement_trading_strategy(
            moe_model, data_dict, all_top_intersection, False, gating_mode, temperature, "ALL_Intersection"
        )
        strategies_results.append(intersection_results)
    
    # treeMoE bottom K
    moe_bottom_results = implement_trading_strategy(moe_model, data_dict, moe_bottom, False, gating_mode, temperature, "MoE_Bottom")
    strategies_results.append(moe_bottom_results)
    
    # XGBoost bottom K
    xgb_bottom_results = implement_trading_strategy(moe_model, data_dict, xgb_bottom, False, gating_mode, temperature, "XGB_Bottom")
    strategies_results.append(xgb_bottom_results)
    
    # CatBoost bottom K
    cat_bottom_results = implement_trading_strategy(moe_model, data_dict, cat_bottom, False, gating_mode, temperature, "CAT_Bottom")
    strategies_results.append(cat_bottom_results)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    performance_df = compare_strategies_performance(strategies_results, f"{output_dir}/multi_model_performance_{timestamp}.png")
    performance_df.to_csv(f"{output_dir}/multi_model_metrics_{timestamp}.csv")
    
    plt.figure(figsize=(15, 10))
    
    def plot_top_features_comparison(ax, top_k=10):
        models = ['MoE', 'XGBoost', 'CatBoost']
        top_features = [moe_top[:top_k], xgb_top[:top_k], cat_top[:top_k]]
        
        all_features = list(set().union(*top_features))
        data = {}
        
        for i, model in enumerate(models):
            data[model] = []
            for feature in all_features:
                if feature in top_features[i]:
                    rank = top_features[i].index(feature) + 1
                    data[model].append(-rank)  # Negative == top K features
                else:
                    data[model].append(0)
        
        df = pd.DataFrame(data, index=all_features)
        df = df.replace(0, np.nan)  
        
        df.plot(kind='barh', ax=ax, width=0.7)
        ax.set_title(f'Top {top_k} Features Comparison Across Models')
        ax.set_xlabel('Feature Rank (Higher = More Important)')
        ax.set_ylabel('Features')
        ax.invert_xaxis()  
        ax.grid(True, axis='x')
        
    def plot_importance_heatmap(ax, top_k=20):
        union_features = list(set().union(moe_top[:top_k], xgb_top[:top_k], cat_top[:top_k]))
        
        data = []
        for feature in union_features:
            moe_idx = np.where(np.array(feature_names) == feature)[0][0] if feature in feature_names else -1
            moe_imp = moe_importance[moe_idx] if moe_idx >= 0 else 0
            
            xgb_idx = np.where(np.array(feature_names) == feature)[0][0] if feature in feature_names else -1
            xgb_imp = xgb_importance[xgb_idx] if xgb_idx >= 0 else 0
            
            cat_idx = np.where(np.array(feature_names) == feature)[0][0] if feature in feature_names else -1
            cat_imp = cat_importance[cat_idx] if cat_idx >= 0 else 0
            
            data.append({
                'Feature': feature,
                'MoE': moe_imp,
                'XGBoost': xgb_imp,
                'CatBoost': cat_imp
            })
        
        df = pd.DataFrame(data).set_index('Feature')
        
        for col in df.columns:
            max_val = df[col].max()
            if max_val > 0:
                df[col] = df[col] / max_val
        
        sns.heatmap(df, cmap='viridis', ax=ax, annot=True, fmt='.2f')
        ax.set_title('Normalized Feature Importance Across Models')
        
    ax1 = plt.subplot(2, 1, 1)
    plot_top_features_comparison(ax1, top_k=plot_top_k)
    
    ax2 = plt.subplot(2, 1, 2)
    plot_importance_heatmap(ax2, top_k=plot_union_top_k)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    moe_imp_dict = {feature_names[i]: moe_importance[i] for i in range(len(feature_names))}
    xgb_imp_dict = {feature_names[i]: xgb_importance[i] for i in range(len(feature_names))}
    cat_imp_dict = {feature_names[i]: cat_importance[i] for i in range(len(feature_names))}
    
    imp_df = pd.DataFrame({
        'Feature': feature_names,
        'MoE': [moe_imp_dict[f] for f in feature_names],
        'XGBoost': [xgb_imp_dict[f] for f in feature_names],
        'CatBoost': [cat_imp_dict[f] for f in feature_names]
    })
    
    corr = imp_df[['MoE', 'XGBoost', 'CatBoost']].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Between Model Feature Importances')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/importance_correlation_{timestamp}.png", dpi=300)
    plt.show()
    
    feature_importance_comparison = {
        'moe_top': moe_top,
        'xgb_top': xgb_top,
        'cat_top': cat_top,
        'moe_bottom': moe_bottom,
        'xgb_bottom': xgb_bottom,
        'cat_bottom': cat_bottom,
        'intersection': all_top_intersection,
        'union': all_top_union,
        'correlation': corr.to_dict()
    }
    
    feature_importance_comparison = convert_numpy_types(feature_importance_comparison)
    
    with open(f"{output_dir}/feature_importance_comparison_{timestamp}.json", 'w') as f:
        json.dump(feature_importance_comparison, f, indent=4)
    
    return {
        'strategies_results': strategies_results,
        'performance_df': performance_df,
        'feature_importance_comparison': feature_importance_comparison
    }




# MEASURE !!
def calculate_returns(signals, close_prices, transaction_cost=0.002):
    """Calculating the rate of return based on the signal"""
    daily_returns = np.zeros(len(signals))
    
    # No profit on the first day as a reference point

    for i in range(1, len(signals)):
        prev_signal = signals[i-1] # Deciding on today's position based on yesterday's signal    
        price_change = (close_prices[i] / close_prices[i-1]) - 1 #Calculate the percentage change in price (Today's price / Yesterday's price - 1)
        
        if prev_signal == 1:  
            daily_returns[i] = price_change   # Buy position
        else:                
            daily_returns[i] = -price_change  # Sell position
        
        if i > 1 and signals[i-1] != signals[i-2]: #Transaction fees apply if there is a change in position
            daily_returns[i] -= transaction_cost
    
    cumulative_returns = np.cumprod(1 + daily_returns) - 1 # Calculate the cumulative return
    
    return daily_returns, cumulative_returns

    


def calculate_performance_metrics(returns, cumulative_returns, signals, y_true):
    # Total return (last data So, if we have starting money, then final return is that starting monty * (1+totral_return)
    total_return = cumulative_returns[-1]
    
    # Annualized return (based on 252 trading days)
    n_days = len(returns)
    annualized_return = (1 + total_return) ** (252 / n_days) - 1
    
    # SRR
    daily_sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)
    
    # MDD
    cumulative_max = np.maximum.accumulate(cumulative_returns)
    drawdown = cumulative_returns - cumulative_max
    max_drawdown = np.min(drawdown)
    
    # win rate
    wins = np.sum(returns > 0)
    win_rate = wins / (len(returns) - 1)  # 첫 날 제외
    
    # profit and loss ratio
    gains = returns[returns > 0].mean() if any(returns > 0) else 0
    losses = abs(returns[returns < 0].mean()) if any(returns < 0) else 1e-9
    profit_loss_ratio = gains / losses if losses != 0 else float('inf')
    
    accuracy = np.mean(signals == y_true)
    
    try:
        precision_buy = np.sum((signals == 1) & (y_true == 1)) / np.sum(signals == 1) if np.sum(signals == 1) > 0 else 0
        precision_sell = np.sum((signals == 0) & (y_true == 0)) / np.sum(signals == 0) if np.sum(signals == 0) > 0 else 0
        
        recall_buy = np.sum((signals == 1) & (y_true == 1)) / np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 0
        recall_sell = np.sum((signals == 0) & (y_true == 0)) / np.sum(y_true == 0) if np.sum(y_true == 0) > 0 else 0
        
        mcc = matthews_corrcoef(y_true, signals)
    except:
        precision_buy, precision_sell = 0, 0
        recall_buy, recall_sell = 0, 0
        mcc = 0
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': daily_sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_loss_ratio': profit_loss_ratio,
        'accuracy': accuracy,
        'precision_buy': precision_buy,
        'precision_sell': precision_sell,
        'recall_buy': recall_buy,
        'recall_sell': recall_sell,
        'mcc': mcc
    }


# Step 4. Variable importance extraction and variable selection functions
def extract_feature_importance(model, data_loader, gating_mode='hard', importance_type='both', top_k= 10, temp = 0.1):
    model.eval()
    
    all_x, all_y, all_dates, all_close = [], [], [], []
    for batch_x, batch_y, batch_close, batch_dates in data_loader:
        all_x.append(batch_x)
        all_y.append(batch_y)
        all_close.append(batch_close)
        all_dates.extend(batch_dates)
    
    X_cat = torch.cat(all_x, dim=0).to(device)
    Y_cat = torch.cat(all_y, dim=0).to(device).squeeze(-1)
    close_prices = torch.cat(all_close, dim=0).cpu().numpy()

    #. Techinical indicator names
    feature_names = data_loader.dataset.feature_names

    # Global selection
    importance_data = {}
    print(f"[INFO] Extracting global feature importance ({gating_mode} gating, {importance_type} importance)...")
    if gating_mode == 'hard':
        if importance_type in ['phi_only', 'both']:
            g_imp_phi, cov_dict_phi = compute_global_importance_hard(model, X_cat, device)
            importance_data['phi_only'] = {'importance': g_imp_phi, 'coverage_dict': cov_dict_phi}
        
        if importance_type in ['gamma_weight', 'both']:
            g_imp_gamma, cov_dict_gamma = compute_global_influence_hard(model, X_cat, device)
            importance_data['gamma_weight'] = {
                'importance': g_imp_gamma,
                'coverage_dict': cov_dict_gamma
            }
    else:  # soft
        temperature = temp
        if importance_type in ['phi_only', 'both']:
            g_imp_phi, cov_dict_phi = compute_global_importance_soft(model, X_cat, device, temperature)
            importance_data['phi_only'] = {
                'importance': g_imp_phi,
                'coverage_dict': cov_dict_phi
            }
        
        if importance_type in ['gamma_weight', 'both']:
            g_imp_gamma, cov_dict_gamma = compute_global_influence_soft(model, X_cat, device, temperature)
            importance_data['gamma_weight'] = {
                'importance': g_imp_gamma,
                'coverage_dict': cov_dict_gamma
            }
    
    # Leaf importance
    print("[INFO] Extracting leaf-level feature importance...")
    leaf_features_data = gather_leaf_feature_selection_data(
        model.root, feature_names, top_k=top_k, bottom_k=top_k
    )
    
    # Impact by leaf  (gamma_weight)
    print("[INFO] Extracting leaf-level feature influence (gamma * weight)...")
    leaf_influence_data = collect_leaf_influence_gamma_weight(
        model=model,
        x=X_cat,
        feature_names=feature_names,
        gating_mode=gating_mode,
        temperature=temp,
        top_k=top_k,
        bottom_k=top_k,
        device=device
    )
    
    return {
        'global_importance': importance_data,
        'leaf_features': leaf_features_data,
        'leaf_influence': leaf_influence_data,
        'feature_names': feature_names,
        'dates': all_dates,
        'close_prices': close_prices,
        'X': X_cat.cpu().numpy(),
        'Y': Y_cat.cpu().numpy()
    }

def run_feature_importance_analysis(model, data_loader, top_k, temp, gating_mode='hard', importance_type='both', output_dir='./results'):
    os.makedirs(output_dir, exist_ok=True)
    
    importance_data = extract_feature_importance( model, data_loader, gating_mode, importance_type, top_k, temp)    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if 'phi_only' in importance_data['global_importance']:
        phi_importance = importance_data['global_importance']['phi_only']['importance']
        phi_df = pd.DataFrame({
            'feature': importance_data['feature_names'],
            'importance': phi_importance
        })
        phi_df = phi_df.sort_values('importance', ascending=False)
        phi_df.to_csv(f"{output_dir}/global_importance_phi_{gating_mode}_{timestamp}.csv", index=False)
    
    if 'gamma_weight' in importance_data['global_importance']:
        gamma_importance = importance_data['global_importance']['gamma_weight']['importance']
        gamma_df = pd.DataFrame({
            'feature': importance_data['feature_names'],
            'importance': gamma_importance
        })
        gamma_df = gamma_df.sort_values('importance', ascending=False)
        gamma_df.to_csv(f"{output_dir}/global_importance_gamma_{gating_mode}_{timestamp}.csv", index=False)
    
    # Leaf
    leaf_df = leaf_selection_to_dataframe(importance_data['leaf_features'])
    leaf_df.to_csv(f"{output_dir}/leaf_importance_{gating_mode}_{timestamp}.csv", index=False)
    
    leaf_infl_df = leaf_influence_to_dataframe(importance_data['leaf_influence'])
    leaf_infl_df.to_csv(f"{output_dir}/leaf_influence_{gating_mode}_{timestamp}.csv", index=False)
    
    print(f"[INFO] Importance analysis results saved to {output_dir}")
    
    return importance_data


def run_trading_strategy_evaluation(model, importance_data, gating_mode='hard', temperature=0.1,
                                   top_k=10, output_dir='./results'):

    # Extract the importance of the entire region
    if 'gamma_weight' in importance_data['global_importance']:
        importance_scores = importance_data['global_importance']['gamma_weight']['importance']
    else:
        importance_scores = importance_data['global_importance']['phi_only']['importance']
    
    feature_names = importance_data['feature_names']
    
    # Various variable selection methods
    top_features = select_features(importance_scores, feature_names, 'top_k', top_k)
    bottom_features = select_features(importance_scores, feature_names, 'bottom_k', top_k)
    random_features = select_features(importance_scores, feature_names, 'random', top_k)
    
    strategies_results = []
    
    all_features_results = implement_trading_strategy(model, importance_data, None, True, gating_mode, temperature, "All_Features")
    strategies_results.append(all_features_results)
    
    top_features_results = implement_trading_strategy(model, importance_data, top_features, False, gating_mode, temperature, "Top_Features")
    strategies_results.append(top_features_results)
    
    # 하위 변수 사용
    bottom_features_results = implement_trading_strategy(model, importance_data, bottom_features, False, gating_mode, temperature, "Bottom_Features")
    strategies_results.append(bottom_features_results)
    
    # 랜덤 변수 사용
    random_features_results = implement_trading_strategy(model, importance_data, random_features, False, gating_mode, temperature, "Random_Features")
    strategies_results.append(random_features_results)
    
    # 성능 비교 및 시각화
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    performance_df = compare_strategies_performance(strategies_results, f"{output_dir}/performance_comparison_{gating_mode}_{timestamp}.png")
    performance_df.to_csv(f"{output_dir}/performance_metrics_{gating_mode}_{timestamp}.csv")
    
    # 변수 상호작용 분석
    interaction_results = analyze_feature_interactions(model, importance_data, top_features, gating_mode, temperature)
    
    # 시장 상황별 분석
    market_conditions = classify_market_conditions(all_features_results['returns'], importance_data['close_prices'])
    market_perf = evaluate_by_market_condition(strategies_results, market_conditions)
    
    # 통계적 유의성 검정
    significance_results = statistical_significance_test(strategies_results)
    
    # 점진적 변수 추가 분석
    incremental_results = evaluate_feature_combinations(model, importance_data, top_features, gating_mode, temperature)
    
    results = {
        'strategies': {strat['name']: {
            'selected_features': strat.get('selected_features'),
            'performance': strat['performance'],
            'total_return': strat['performance']['total_return'],
            'sharpe_ratio': strat['performance']['sharpe_ratio']
        } for strat in strategies_results},
        'feature_interactions': {
            'top_synergy_pairs': interaction_results['top_synergy_pairs']
        },
        'market_conditions': {
            cond: {strat: {'avg_return': data['avg_return']} 
                  for strat, data in strats.items()}
            for cond, strats in market_perf.items()
        },
        'statistical_significance': {
            'significant_comparisons': {
                comp: data for comp, data in significance_results['t_tests'].items() 
                if data['significant']
            }
        }
    }
    
    results_converted = convert_numpy_types(results)
    
    # JSON으로 저장
    with open(f"{output_dir}/trading_evaluation_{gating_mode}_{timestamp}.json", 'w') as f:
        json.dump(results_converted, f, indent=4)
    
    print(f"[INFO] Trading strategy evaluation results saved to {output_dir}")
    
    return {
        'strategies_results': strategies_results,
        'interaction_results': interaction_results,
        'market_performance': market_perf,
        'significance_results': significance_results,
        'incremental_results': incremental_results
    }
