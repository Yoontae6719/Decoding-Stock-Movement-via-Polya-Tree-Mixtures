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

from xai_utils_ver_a.utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 

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
                                   top_k=10, plot_top_k=10, plot_union_top_k = 10, output_dir='./XAI_results'):
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





























