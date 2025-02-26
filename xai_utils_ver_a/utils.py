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


















































































































