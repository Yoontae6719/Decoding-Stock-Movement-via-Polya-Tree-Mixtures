from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import treeMoE

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import os
import time
import json

import warnings
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from utils.tools import EarlyStopping, adjust_learning_rate, TemperatureScheduler
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import shutil
import gc

class Exp_Trading(Exp_Basic):
    def __init__(self, args):
        super(Exp_Trading, self).__init__(args)
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': []}
        
    def _build_model(self):
        model_dict = {
            'treeMoE': treeMoE,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_gpu:
            model = model.to(self.device)
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def fit(self, setting):
        sched = TemperatureScheduler(
            initial_temp=self.args.initial_temp,
            final_temp=self.args.final_temp,
            anneal_epochs=self.args.anneal_epochs,
            schedule_type=self.args.schedule_type,
        )

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')        
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        outputs_path = f'./trading/trading_{self.args.data}/{setting}/'

        if not os.path.exists(outputs_path):
            os.makedirs(outputs_path)

        model_optim = self._select_optimizer()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        train_size = len(train_data)  

        best_vali_loss = float('inf')
        best_metrics = {}
        best_temp = None

        for epoch in range(self.args.train_epochs):
            self.model.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            train_preds = []
            train_labels = []

            current_temp = sched.get_temp(epoch) if self.args.temperature_scheduler else 0.5

            for x_batch, y_batch, x_close in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device).squeeze(-1)
                batch_size = x_batch.size(0)
                
                model_optim.zero_grad()

                probs = self.model(
                    x_batch,
                    temperature_fs=current_temp,  
                    temperature_gate=0.5,      
                )

                nll_loss = -torch.log(probs[torch.arange(batch_size), y_batch] + 1e-8).mean()

                kl_loss = self.model.get_kl_loss()

                kl_loss_scaled = kl_loss * (batch_size / train_size)

                loss = nll_loss + self.args.lambda_KL * kl_loss_scaled

                loss.backward()
                if self.args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                model_optim.step()

                total_loss += loss.item() * batch_size
                total_samples += batch_size
                preds = torch.argmax(probs, dim=1)
                total_correct += (preds == y_batch).sum().item()

                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(y_batch.cpu().numpy())

            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples
            
            train_mcc = matthews_corrcoef(train_labels, train_preds)
            
            vali_loss, vali_acc, vali_mcc, vali_f1 = self.evaluate(vali_loader, current_temp)
            test_loss, test_acc, test_mcc, test_f1 = self.evaluate(test_loader, current_temp)

            print(f"[Epoch {epoch+1}/{self.args.train_epochs}] "
                f"Temp={current_temp:.3f} "
                f"train_loss={avg_loss:.4f}, train_acc={avg_acc:.4f}, train_mcc={train_mcc:.4f}\n"
                f"             valid_loss={vali_loss:.4f}, valid_acc={vali_acc:.4f}, valid_mcc={vali_mcc:.4f}, valid_f1={vali_f1:.4f}\n"
                f"             test_loss={test_loss:.4f}, test_acc={test_acc:.4f}, test_mcc={test_mcc:.4f}, test_f1={test_f1:.4f}")

            if vali_loss < best_vali_loss:
                best_vali_loss = vali_loss
                best_metrics = {
                    'vali_loss': vali_loss,
                    'vali_acc': vali_acc,
                    'vali_mcc': vali_mcc,
                    'vali_f1' : vali_f1,
                    'test_loss': test_loss,
                    'test_acc': test_acc,
                    'test_mcc': test_mcc,
                    'test_f1' : test_f1,
                    "current_temp": current_temp,
                }
                best_temp = current_temp

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                df = pd.DataFrame([best_metrics])  
                df.to_csv(os.path.join(outputs_path, 'best_metrics.csv'), index=False)
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)
        
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        with open(os.path.join(outputs_path, 'best_metrics.json'), 'w') as f:
            json.dump(best_metrics, f, indent=2)
            
        print(f"Best temperature: {best_temp} with validation loss: {best_vali_loss:.4f}")
        
        return best_temp, best_metrics
    
    def evaluate(self, data_loader, current_temp):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        preds_list = []
        labels_list = []

        with torch.no_grad():
            for x_batch, y_batch, x_close in data_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device).squeeze(-1)

                probs = self.model(x_batch, temperature_fs=current_temp, temperature_gate=0.5)

                nll = -torch.log(probs[torch.arange(x_batch.size(0)), y_batch] + 1e-8).sum().item()

                preds = torch.argmax(probs, dim=1)
                correct = (preds == y_batch).sum().item()
                
                total_loss += nll
                total_correct += correct
                total_samples += x_batch.size(0)

                preds_list.extend(preds.cpu().numpy())
                labels_list.extend(y_batch.cpu().numpy())

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        mcc = matthews_corrcoef(labels_list, preds_list)
        f1_macro = f1_score(labels_list, preds_list, average='macro')

        return avg_loss, avg_acc, mcc, f1_macro
    
    def trading(self, setting):
        print('Starting trading analysis...')
        
        best_temp, best_metrics = self.fit(setting)
        
        outputs_path = f'./trading/trading_{self.args.data}/{setting}/'
        if not os.path.exists(outputs_path):
            os.makedirs(outputs_path)
        
        analysis_path = f'./trading/analysis/{self.args.data}/'
        if not os.path.exists(analysis_path):
            os.makedirs(analysis_path)
        
        print('Loading data...')
        train_data, train_loader = self._get_data(flag='train')
        test_data, test_loader = self._get_data(flag='test')
        
        if hasattr(test_data, 'feature_names'):
            self.feature_names = test_data.feature_names
        else:
            self.feature_names = [f"Feature_{i}" for i in range(self.args.dim_input)]
        
        print(f'Analyzing tree structure using temperature: {best_temp}...')
        node_statistics, feature_importance = self.analyze_tree_structure(
            self.model.root, "root", best_temp)
        
        print('Analyzing node statistics...')
        self.analyze_node_statistics(test_loader, node_statistics, feature_importance, best_temp)
        
        print('Calculating importance metrics...')
        global_importance = self.calculate_global_feature_importance(node_statistics, feature_importance)
        
        feature_selection_freq = self.calculate_feature_selection_frequency(feature_importance)
        phi_importance = self.calculate_phi_importance(feature_importance)
        coverage_weighted_importance = self.calculate_coverage_weighted_importance(node_statistics, feature_importance)
        
        all_importance_metrics = {
            'global_importance': global_importance,
            'feature_selection_frequency': feature_selection_freq,
            'phi_importance': phi_importance,
            'coverage_weighted_importance': coverage_weighted_importance
        }
        
        print('Extracting important features for each leaf...')
        leaf_features = self.calculate_leaf_feature_importance(node_statistics, feature_importance)
        
        with open(os.path.join(outputs_path, f"{self.args.model_id}_xai_features.json"), 'w') as f:
            json.dump(leaf_features, f, indent=2)
        
        for metric_name, values in all_importance_metrics.items():
            df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': values
            })
            df = df.sort_values('importance', ascending=False)
            df.to_csv(os.path.join(outputs_path, f"{metric_name}.csv"), index=False)
        
        print('Converting data loaders to dataframes...')
        train_df = self.prepare_dataset_from_loader(train_loader)
        test_df = self.prepare_dataset_from_loader(test_loader)
        
        if 'Date' not in train_df.columns:
            train_dates = pd.date_range(start='2020-01-01', periods=len(train_df))
            train_df['Date'] = train_dates
            
        if 'Date' not in test_df.columns:
            test_dates = pd.date_range(start='2023-01-01', periods=len(test_df))
            test_df['Date'] = test_dates
        
        top_k = self.args.top_k if hasattr(self.args, 'top_k') else 5
        print(f"Using top_k={top_k} features for all strategies")
        
        print('Running trading backtest...')
        strategy_results = []
        
        for leaf_path, leaf_data in leaf_features.items():
            if leaf_data['node_stats'].get('reliability', 'low') == 'low':
                print(f"\nSkipping leaf {leaf_path} due to low reliability")
                continue
                
            important_features = leaf_data['important']['features']
            important_scores = leaf_data['important']['importance_scores']
            
            important_weights = {important_features[i]: important_scores[i] 
                                for i in range(len(important_features))}
            
            print(f"\nLeaf {leaf_path}: Training model with {len(important_features)} important features...")
            
            if important_features:
                important_model = self.train_strategy_model(train_df, important_features, important_weights)
                important_backtest = self.backtest_strategy(important_model, test_df)
                
                important_result = {
                    'Stock': self.args.model_id,
                    'Strategy': f"Leaf: {leaf_path}, Important",
                    'Leaf': leaf_path,
                    'Feature_Type': 'Important',
                    'Features': ','.join(important_features),
                    'Accuracy': important_backtest['accuracy'],
                    'F1_Score': important_backtest['f1_score'],
                    'MCC': important_backtest['mcc'],
                    'Sharpe_Ratio': important_backtest['sharpe_ratio'],
                    'Max_Drawdown': important_backtest['max_drawdown'],
                    'Final_Return': important_backtest['final_return'],
                    'Coverage': leaf_data['node_stats']['coverage'],
                    'Leaf_Accuracy': leaf_data['node_stats']['accuracy'],
                    'Leaf_MCC': leaf_data['node_stats']['mcc'],
                    'Reliability': leaf_data['node_stats'].get('reliability', 'medium')
                }
                strategy_results.append(important_result)
            
            unimportant_features = leaf_data['unimportant']['features']
            unimportant_scores = leaf_data['unimportant']['importance_scores']
            
            unimportant_weights = {unimportant_features[i]: unimportant_scores[i] 
                                for i in range(len(unimportant_features))}
            
            print(f"Leaf {leaf_path}: Training model with {len(unimportant_features)} unimportant features...")
            
            if unimportant_features:
                unimportant_model = self.train_strategy_model(train_df, unimportant_features, unimportant_weights)
                unimportant_backtest = self.backtest_strategy(unimportant_model, test_df)
                
                unimportant_result = {
                    'Stock': self.args.model_id,
                    'Strategy': f"Leaf: {leaf_path}, Unimportant",
                    'Leaf': leaf_path,
                    'Feature_Type': 'Unimportant',
                    'Features': ','.join(unimportant_features),
                    'Accuracy': unimportant_backtest['accuracy'],
                    'F1_Score': unimportant_backtest['f1_score'],
                    'MCC': unimportant_backtest['mcc'],
                    'Sharpe_Ratio': unimportant_backtest['sharpe_ratio'],
                    'Max_Drawdown': unimportant_backtest['max_drawdown'],
                    'Final_Return': unimportant_backtest['final_return'],
                    'Coverage': leaf_data['node_stats']['coverage'],
                    'Leaf_Accuracy': leaf_data['node_stats']['accuracy'],
                    'Leaf_MCC': leaf_data['node_stats']['mcc'],
                    'Reliability': leaf_data['node_stats'].get('reliability', 'medium')
                }
                strategy_results.append(unimportant_result)
        
        print("\nRunning global importance-based trading strategies...")
        additional_results = self.run_global_importance_strategies(
            train_df, test_df, all_importance_metrics, top_k, analysis_path)
        
        strategy_results.extend(additional_results)
        
        results_df = pd.DataFrame(strategy_results)
        results_path = os.path.join(outputs_path, f"{self.args.model_id}_trading_results.csv")
        results_df.to_csv(results_path, index=False)
        
        analysis_results_path = os.path.join(analysis_path, f"{self.args.model_id}_analysis_results.csv")
        analysis_results = [r for r in strategy_results if not r['Strategy'].startswith("Leaf:")]
        analysis_df = pd.DataFrame(analysis_results)
        analysis_df.to_csv(analysis_results_path, index=False)
        
        print("\nTrading results summary:")
        if 'Reliability' in results_df.columns:
            summary = results_df.groupby(['Feature_Type', 'Reliability']).agg({
                'Accuracy': ['mean'],
                'MCC': ['mean'],
                'Sharpe_Ratio': ['mean'],
                'Final_Return': ['mean']
            })
        else:
            summary = results_df.groupby('Feature_Type').agg({
                'Accuracy': ['mean'],
                'MCC': ['mean'],
                'Sharpe_Ratio': ['mean'],
                'Final_Return': ['mean']
            })
        print(summary)
        
        self.plot_analysis_summary(analysis_df, analysis_path)
        
        return {
            'node_statistics': node_statistics,
            'feature_importance': feature_importance,
            'leaf_features': leaf_features,
            'strategy_results': results_df,
            'best_temp': best_temp,
            'best_metrics': best_metrics,
            'importance_metrics': all_importance_metrics
        }
    
    def analyze_tree_structure(self, node, node_name, current_temp):
        node_statistics = {}
        feature_importance = {}
        
        if node.is_leaf:
            if node.use_feature_selection and node.fs_module is not None:
                with torch.no_grad():
                    gamma, phi = node.fs_module(temperature=current_temp, training=False)
                    phi = phi.detach().cpu().numpy()
                
                fc1_weights = node.local_expert.fc1.weight.detach().cpu().numpy()
                
                alpha = np.mean(np.abs(fc1_weights), axis=0)
                
                feature_importance[node_name] = {
                    'phi': phi,
                    'alpha': alpha,
                    'importance': phi * alpha,
                    'node_type': 'leaf',
                    'depth': node.depth
                }
            else:
                alpha = np.ones(self.args.dim_input)
                
                if hasattr(node, 'local_expert') and hasattr(node.local_expert, 'fc1'):
                    fc1_weights = node.local_expert.fc1.weight.detach().cpu().numpy()
                    alpha = np.mean(np.abs(fc1_weights), axis=0)
                    
                feature_importance[node_name] = {
                    'phi': np.ones(self.args.dim_input),
                    'alpha': alpha,
                    'importance': alpha,
                    'node_type': 'leaf',
                    'depth': node.depth
                }
                
            node_statistics[node_name] = {
                'type': 'leaf',
                'depth': node.depth,
                'coverage': 0,
                'samples': 0,
                'class_distribution': np.zeros(self.args.num_classes),
                'accuracy': 0,
                'mcc': 0
            }
        else:
            if node.use_feature_selection and node.fs_module is not None:
                with torch.no_grad():
                    gamma, phi = node.fs_module(temperature=current_temp, training=False)
                    phi = phi.detach().cpu().numpy()
            else:
                phi = np.ones(self.args.dim_input)
                
            if node.gating is not None:
                gating_weights = node.gating[0].weight.detach().cpu().numpy()
                alpha = np.mean(np.abs(gating_weights), axis=0)
            else:
                alpha = np.abs(node.w.detach().cpu().numpy())
                
            feature_importance[node_name] = {
                'phi': phi,
                'alpha': alpha,
                'importance': phi * alpha,
                'node_type': 'internal',
                'depth': node.depth
            }
            
            node_statistics[node_name] = {
                'type': 'internal',
                'depth': node.depth,
                'coverage': 0,
                'samples': 0,
                'left_prob': 0,
                'right_prob': 0,
                'class_distribution': np.zeros(self.args.num_classes)
            }
            
            child_stats, child_imp = self.analyze_tree_structure(node.left_child, f"{node_name}_L", current_temp)
            node_statistics.update(child_stats)
            feature_importance.update(child_imp)
            
            child_stats, child_imp = self.analyze_tree_structure(node.right_child, f"{node_name}_R", current_temp)
            node_statistics.update(child_stats)
            feature_importance.update(child_imp)
            
        return node_statistics, feature_importance
    
    def analyze_node_statistics(self, data_loader, node_statistics, feature_importance, current_temp):
        node_samples = defaultdict(int)
        node_classes = defaultdict(lambda: np.zeros(self.args.num_classes))
        node_correct = defaultdict(int)
        node_predictions = defaultdict(list)
        node_true_labels = defaultdict(list)
        
        total_samples = 0
        
        if current_temp is None:
            current_temp = 0.5
            print("Warning: Temperature was None, using default value of 0.5")
            
        self.model.eval()
        with torch.no_grad():
            for x_batch, y_batch, _ in data_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device).squeeze(-1)
                batch_size = x_batch.size(0)
                total_samples += batch_size
                
                for i in range(batch_size):
                    x = x_batch[i:i+1]
                    label = y_batch[i].item()
                    
                    current_node = self.model.root
                    node_path = ["root"]
                    node_samples["root"] += 1
                    node_classes["root"][label] += 1
                    
                    while not current_node.is_leaf:
                        if current_node.use_feature_selection:
                            phi = torch.sigmoid(current_node.fs_module.logit_phi)
                            gamma = (phi > 0.5).float()
                            x_masked = x * gamma.unsqueeze(0)
                        else:
                            x_masked = x
                        
                        if current_node.gating is not None:
                            gate_logit = current_node.gating(x_masked).squeeze(-1)
                        else:
                            gate_logit = (x_masked * current_node.w).sum(dim=1) + current_node.b
                        
                        temp_value = 0.5
                        p_left = torch.sigmoid(gate_logit / temp_value)
                        go_left = (p_left > 0.5).item()
                        
                        if go_left:
                            current_node = current_node.left_child
                            node_path.append(f"{node_path[-1]}_L")
                        else:
                            current_node = current_node.right_child
                            node_path.append(f"{node_path[-1]}_R")
                        
                        node_samples[node_path[-1]] += 1
                        node_classes[node_path[-1]][label] += 1
                    
                    leaf_name = node_path[-1]
                    
                    pred_probs = self.model(x, temperature_fs=current_temp, temperature_gate=0.5)
                    pred = torch.argmax(pred_probs, dim=1).item()
                    
                    node_predictions[leaf_name].append(pred)
                    node_true_labels[leaf_name].append(label)
                    if pred == label:
                        node_correct[leaf_name] += 1
        
        for node_name in node_statistics:
            coverage = node_samples[node_name] / total_samples if total_samples > 0 else 0
            node_statistics[node_name]['coverage'] = coverage
            node_statistics[node_name]['samples'] = node_samples[node_name]
            node_statistics[node_name]['class_distribution'] = node_classes[node_name]
            
            if node_statistics[node_name]['type'] == 'leaf':
                if node_samples[node_name] > 0:
                    accuracy = node_correct[node_name] / node_samples[node_name]
                    
                    if len(set(node_true_labels[node_name])) > 1 and len(node_true_labels[node_name]) > 1:
                        mcc = matthews_corrcoef(node_true_labels[node_name], node_predictions[node_name])
                    else:
                        mcc = 0
                    
                    node_statistics[node_name]['accuracy'] = accuracy
                    node_statistics[node_name]['mcc'] = mcc
                    node_statistics[node_name]['predictions'] = node_predictions[node_name]
                    node_statistics[node_name]['true_labels'] = node_true_labels[node_name]
    
    def calculate_global_feature_importance(self, node_statistics, feature_importance):
        global_importance = np.zeros(self.args.dim_input)
        total_weight = 0
        
        for node_name, importance in feature_importance.items():
            node_stats = node_statistics[node_name]
            node_coverage = node_stats['coverage']
            sample_count = node_stats['samples']
            
            if sample_count == 0:
                continue
            
            sample_weight = np.sqrt(sample_count)
            
            if node_stats['type'] == 'leaf':
                node_accuracy = node_stats.get('accuracy', 0.5)
                node_mcc = node_stats.get('mcc', 0)
                
                reliability = max(0, abs(node_accuracy - 0.5) * 2)
                
                node_weight = node_coverage * sample_weight * (reliability + 0.5) * (node_accuracy * (1 + node_mcc) / 2)
            else:
                node_weight = node_coverage * sample_weight
            
            global_importance += node_weight * importance['importance']
            total_weight += node_weight
        
        if total_weight > 0:
            global_importance = global_importance / total_weight
        
        return global_importance
    
    def calculate_feature_selection_frequency(self, feature_importance):
        feature_selection_freq = np.zeros(self.args.dim_input)
        total_nodes = 0
        
        for node_name, importance in feature_importance.items():
            phi = importance['phi']
            selected_features = (phi > 0.5).astype(float)
            feature_selection_freq += selected_features
            total_nodes += 1
        
        if total_nodes > 0:
            feature_selection_freq /= total_nodes
        
        return feature_selection_freq
    
    def calculate_phi_importance(self, feature_importance):
        phi_importance = np.zeros(self.args.dim_input)
        total_nodes = 0
        
        for node_name, importance in feature_importance.items():
            phi = importance['phi']
            phi_importance += phi
            total_nodes += 1
        
        if total_nodes > 0:
            phi_importance /= total_nodes
        
        return phi_importance
    
    def calculate_coverage_weighted_importance(self, node_statistics, feature_importance):
        coverage_weighted_importance = np.zeros(self.args.dim_input)
        total_weight = 0
        
        for node_name, importance in feature_importance.items():
            node_stats = node_statistics[node_name]
            coverage = node_stats['coverage']
            sample_count = node_stats['samples']
            
            if sample_count == 0:
                continue
                
            sample_weight = np.sqrt(sample_count)
            node_weight = coverage * sample_weight
            
            coverage_weighted_importance += node_weight * importance['importance']
            total_weight += node_weight
        
        if total_weight > 0:
            coverage_weighted_importance /= total_weight
        
        return coverage_weighted_importance
    
    def calculate_leaf_feature_importance(self, node_statistics, feature_importance):
        leaf_features = {}
        top_k = self.args.top_k if hasattr(self.args, 'top_k') else 5
        
        for node_name, importance in feature_importance.items():
            if node_statistics[node_name]['type'] != 'leaf':
                continue
            
            sample_count = node_statistics[node_name]['samples']
            
            if sample_count < 10:
                imp_scores = importance['importance']
                
                top_indices = np.argsort(imp_scores)[-top_k:]
                top_features = [self.feature_names[i] for i in top_indices]
                top_importance = imp_scores[top_indices]
                
                bottom_indices = np.argsort(imp_scores)[:top_k]
                bottom_features = [self.feature_names[i] for i in bottom_indices]
                bottom_importance = imp_scores[bottom_indices]
                
                leaf_features[node_name] = {
                    'important': {
                        'features': top_features,
                        'importance_scores': top_importance.tolist(),
                        'reliability': 'low'
                    },
                    'unimportant': {
                        'features': bottom_features,
                        'importance_scores': bottom_importance.tolist(),
                        'reliability': 'low'
                    },
                    'node_stats': {
                        'coverage': node_statistics[node_name]['coverage'],
                        'accuracy': node_statistics[node_name]['accuracy'],
                        'mcc': node_statistics[node_name]['mcc'],
                        'sample_count': sample_count,
                        'depth': node_statistics[node_name]['depth'],
                        'reliability': 'low'
                    }
                }
                continue
                
            imp_scores = importance['importance']
            
            node_accuracy = node_statistics[node_name].get('accuracy', 0.5)
            node_mcc = node_statistics[node_name].get('mcc', 0)
            
            reliability = min(1.0, sample_count / 100) * (abs(node_accuracy - 0.5) * 2) * (node_mcc + 1) / 2
            reliability_level = 'high' if reliability > 0.5 else 'medium' if reliability > 0.2 else 'low'
            
            top_indices = np.argsort(imp_scores)[-top_k:]
            top_features = [self.feature_names[i] for i in top_indices]
            top_importance = imp_scores[top_indices]
            
            bottom_indices = np.argsort(imp_scores)[:top_k]
            bottom_features = [self.feature_names[i] for i in bottom_indices]
            bottom_importance = imp_scores[bottom_indices]
            
            leaf_features[node_name] = {
                'important': {
                    'features': top_features,
                    'importance_scores': top_importance.tolist(),
                    'reliability': reliability_level
                },
                'unimportant': {
                    'features': bottom_features,
                    'importance_scores': bottom_importance.tolist(),
                    'reliability': reliability_level
                },
                'node_stats': {
                    'coverage': node_statistics[node_name]['coverage'],
                    'accuracy': node_statistics[node_name]['accuracy'],
                    'mcc': node_statistics[node_name]['mcc'],
                    'sample_count': sample_count,
                    'depth': node_statistics[node_name]['depth'],
                    'reliability': reliability_level,
                    'reliability_score': reliability
                }
            }
        
        return leaf_features
    
    def prepare_dataset_from_loader(self, data_loader):
        all_features = []
        all_labels = []
        all_prices = []
        
        for x_batch, y_batch, close_batch in data_loader:
            all_features.append(x_batch.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
            all_prices.append(close_batch.cpu().numpy())
        
        if all_features:
            features = np.vstack(all_features)
            labels = np.vstack(all_labels).squeeze()
            prices = np.concatenate(all_prices)
            
            df = pd.DataFrame(features, columns=self.feature_names)
            
            df['y_numeric'] = labels
            df['Y'] = df['y_numeric'].map({0: 'SELL', 1: 'BUY'})
            df['Close'] = prices
            
            dates = pd.date_range(
                start='2020-01-01' if 'train' in str(data_loader.dataset) else '2023-01-01', 
                periods=len(df)
            )
            df['Date'] = dates
            
            return df
        
        return pd.DataFrame()
    
    def train_strategy_model(self, train_df, features, importance_weights=None):
        valid_features = [f for f in features if f in train_df.columns]
        
        if len(valid_features) < 1:
            raise ValueError("No valid features for training")
            
        X = train_df[valid_features].values
        y = train_df['y_numeric'].values
        
        if importance_weights is not None:
            weights = np.array([importance_weights.get(f, 1.0) for f in valid_features])
            
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights) * len(weights)
            else:
                weights = np.ones(len(valid_features))
                
            X_weighted = X * weights.reshape(1, -1)
        else:
            X_weighted = X
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_weighted)
        
        model = LogisticRegression(max_iter=10000, random_state=42, class_weight='balanced')
        model.fit(X_scaled, y)
        
        return {
            'model': model,
            'scaler': scaler,
            'features': valid_features,
            'importance_weights': weights if importance_weights is not None else None
        }
    
    def backtest_strategy(self, strategy_model, test_df, target_col='y_numeric'):
        model = strategy_model['model']
        scaler = strategy_model['scaler']
        features = strategy_model['features']
        importance_weights = strategy_model.get('importance_weights', None)
        
        X = test_df[features].values
        y_true = test_df[target_col].values
        
        if importance_weights is not None:
            X_weighted = X * importance_weights.reshape(1, -1)
        else:
            X_weighted = X
        
        X_scaled = scaler.transform(X_weighted)
        
        y_pred = model.predict(X_scaled)
        
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        mcc = matthews_corrcoef(y_true, y_pred)
        
        prices = test_df['Close'].values
        
        portfolio_value = np.ones(len(prices))
        positions = np.zeros(len(prices))
        
        holding = False
        entry_price = 0
        entry_idx = 0
        trades = []
        
        for i in range(len(prices) - 1):
            current_price = prices[i]
            next_price = prices[i + 1]
            
            if y_pred[i] == 1:
                if not holding:
                    holding = True
                    entry_price = current_price
                    entry_idx = i
                    trades.append({
                        'type': 'BUY',
                        'idx': i,
                        'price': current_price
                    })
                
                portfolio_value[i + 1] = portfolio_value[i] * (next_price / current_price)
                positions[i + 1] = 1
            
            else:
                if holding:
                    exit_return = (current_price - entry_price) / entry_price
                    trades.append({
                        'type': 'SELL',
                        'idx': i,
                        'price': current_price,
                        'return': exit_return,
                        'holding_period': i - entry_idx
                    })
                    holding = False
                
                portfolio_value[i + 1] = portfolio_value[i]
                positions[i + 1] = 0
        
        if holding:
            final_price = prices[-1]
            final_return = (final_price - entry_price) / entry_price
            trades.append({
                'type': 'FINAL_CLOSE',
                'idx': len(prices) - 1,
                'price': final_price,
                'return': final_return,
                'holding_period': len(prices) - 1 - entry_idx
            })
            
        daily_returns = np.zeros(len(portfolio_value) - 1)
        for i in range(len(daily_returns)):
            daily_returns[i] = portfolio_value[i + 1] / portfolio_value[i] - 1
        
        cum_returns = np.cumprod(1 + daily_returns) - 1
        
        if len(daily_returns) > 0 and np.std(daily_returns) > 0:
            sharpe_ratio = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns)
        else:
            sharpe_ratio = 0
        
        peak = np.maximum.accumulate(portfolio_value)
        drawdown = portfolio_value / peak - 1
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        final_return = portfolio_value[-1] / portfolio_value[0] - 1
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'mcc': mcc,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_return': final_return,
            'portfolio_value': portfolio_value,
            'positions': positions,
            'predictions': y_pred,
            'actual': y_true,
            'dates': test_df['Date'].values,
            'trades': trades
        }
    
    def run_global_importance_strategies(self, train_df, test_df, importance_metrics, top_k, output_dir):
        results = []
        
        for metric_name, importance_values in importance_metrics.items():
            feature_importance_dict = {self.feature_names[i]: float(importance_values[i]) 
                                    for i in range(len(self.feature_names))}
            
            top_indices = np.argsort(importance_values)[-top_k:]
            top_features = [self.feature_names[i] for i in top_indices]
            
            top_importance_weights = {f: feature_importance_dict[f] for f in top_features}
            
            bottom_indices = np.argsort(importance_values)[:top_k]
            bottom_features = [self.feature_names[i] for i in bottom_indices]
            
            bottom_importance_weights = {f: feature_importance_dict[f] for f in bottom_features}
            
            mid_point = len(importance_values) // 2
            mid_start = mid_point - top_k // 2
            mid_indices = np.argsort(importance_values)[mid_start:mid_start+top_k]
            mid_features = [self.feature_names[i] for i in mid_indices]
            
            mid_importance_weights = {f: feature_importance_dict[f] for f in mid_features}
            
            random_indices = np.random.choice(len(importance_values), top_k, replace=False)
            random_features = [self.feature_names[i] for i in random_indices]
            
            random_importance_weights = {f: 1.0 for f in random_features}
            
            print(f"\nTraining model with top {top_k} features by {metric_name}...")
            if top_features:
                top_model = self.train_strategy_model(train_df, top_features, top_importance_weights)
                top_backtest = self.backtest_strategy(top_model, test_df)
                
                top_result = {
                    'Stock': self.args.model_id,
                    'Strategy': f"{metric_name}_top{top_k}",
                    'Leaf': '',
                    'Feature_Type': f'Top {top_k}',
                    'Features': ','.join(top_features),
                    'Accuracy': top_backtest['accuracy'],
                    'F1_Score': top_backtest['f1_score'],
                    'MCC': top_backtest['mcc'],
                    'Sharpe_Ratio': top_backtest['sharpe_ratio'],
                    'Max_Drawdown': top_backtest['max_drawdown'],
                    'Final_Return': top_backtest['final_return'],
                    'Coverage': 1.0,
                    'Leaf_Accuracy': top_backtest['accuracy'],
                    'Leaf_MCC': top_backtest['mcc'],
                    'Reliability': 'high'
                }
                results.append(top_result)
            
            print(f"Training model with bottom {top_k} features by {metric_name}...")
            if bottom_features:
                bottom_model = self.train_strategy_model(train_df, bottom_features, bottom_importance_weights)
                bottom_backtest = self.backtest_strategy(bottom_model, test_df)
                
                bottom_result = {
                    'Stock': self.args.model_id,
                    'Strategy': f"{metric_name}_bottom{top_k}",
                    'Leaf': '',
                    'Feature_Type': f'Bottom {top_k}',
                    'Features': ','.join(bottom_features),
                    'Accuracy': bottom_backtest['accuracy'],
                    'F1_Score': bottom_backtest['f1_score'],
                    'MCC': bottom_backtest['mcc'],
                    'Sharpe_Ratio': bottom_backtest['sharpe_ratio'],
                    'Max_Drawdown': bottom_backtest['max_drawdown'],
                    'Final_Return': bottom_backtest['final_return'],
                    'Coverage': 1.0,
                    'Leaf_Accuracy': bottom_backtest['accuracy'],
                    'Leaf_MCC': bottom_backtest['mcc'],
                    'Reliability': 'high'
                }
                results.append(bottom_result)
                
            print(f"Training model with middle {top_k} features by {metric_name}...")
            if mid_features:
                mid_model = self.train_strategy_model(train_df, mid_features, mid_importance_weights)
                mid_backtest = self.backtest_strategy(mid_model, test_df)
                
                mid_result = {
                    'Stock': self.args.model_id,
                    'Strategy': f"{metric_name}_mid{top_k}",
                    'Leaf': '',
                    'Feature_Type': f'Middle {top_k}',
                    'Features': ','.join(mid_features),
                    'Accuracy': mid_backtest['accuracy'],
                    'F1_Score': mid_backtest['f1_score'],
                    'MCC': mid_backtest['mcc'],
                    'Sharpe_Ratio': mid_backtest['sharpe_ratio'],
                    'Max_Drawdown': mid_backtest['max_drawdown'],
                    'Final_Return': mid_backtest['final_return'],
                    'Coverage': 1.0,
                    'Leaf_Accuracy': mid_backtest['accuracy'],
                    'Leaf_MCC': mid_backtest['mcc'],
                    'Reliability': 'high'
                }
                results.append(mid_result)
                
            print(f"Training model with random {top_k} features (baseline)...")
            if random_features:
                random_model = self.train_strategy_model(train_df, random_features, random_importance_weights)
                random_backtest = self.backtest_strategy(random_model, test_df)
                
                random_result = {
                    'Stock': self.args.model_id,
                    'Strategy': f"random{top_k}",
                    'Leaf': '',
                    'Feature_Type': f'Random {top_k}',
                    'Features': ','.join(random_features),
                    'Accuracy': random_backtest['accuracy'],
                    'F1_Score': random_backtest['f1_score'],
                    'MCC': random_backtest['mcc'],
                    'Sharpe_Ratio': random_backtest['sharpe_ratio'],
                    'Max_Drawdown': random_backtest['max_drawdown'],
                    'Final_Return': random_backtest['final_return'],
                    'Coverage': 1.0,
                    'Leaf_Accuracy': random_backtest['accuracy'],
                    'Leaf_MCC': random_backtest['mcc'],
                    'Reliability': 'high'
                }
                results.append(random_result)
        
        return results
    
    def plot_analysis_summary(self, results_df, output_dir):
        if results_df.empty:
            return
            
        strategies = [strat for strat in results_df['Strategy'].unique() if not strat.startswith('Leaf:')]
        
        if not strategies:
            return
            
        metrics = ['Accuracy', 'MCC', 'Sharpe_Ratio', 'Final_Return']
        
        plt.figure(figsize=(15, 12))
        for i, metric in enumerate(metrics):
            plt.subplot(2, 2, i+1)
            
            strategies_list = []
            metric_values = []
            categories = []
            
            for strategy in strategies:
                strat_df = results_df[results_df['Strategy'] == strategy]
                if not strat_df.empty:
                    strategies_list.append(strategy)
                    metric_values.append(strat_df[metric].values[0])
                    
                    if 'top' in strategy:
                        categories.append('top')
                    elif 'mid' in strategy:
                        categories.append('mid')
                    elif 'bottom' in strategy:
                        categories.append('bottom')
                    elif 'random' in strategy:
                        categories.append('random')
                    else:
                        categories.append('other')
            
            plot_df = pd.DataFrame({
                'Strategy': strategies_list,
                'Value': metric_values,
                'Category': categories
            })
            
            plot_df = plot_df.sort_values(['Category', 'Value'], ascending=[True, False])
            
            colors = []
            for cat in plot_df['Category']:
                if cat == 'top':
                    colors.append('green')
                elif cat == 'mid':
                    colors.append('blue')
                elif cat == 'bottom':
                    colors.append('red')
                elif cat == 'random':
                    colors.append('gray')
                else:
                    colors.append('purple')
            
            plt.bar(plot_df['Strategy'], plot_df['Value'], color=colors)
            plt.xticks(rotation=45, ha='right')
            plt.ylabel(f'Average {metric}')
            plt.title(f'Comparison of Strategies by {metric}')
            plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f"{self.args.data}_analysis_summary.png"))
        plt.close()
        
        plt.figure(figsize=(15, 12))
        
        metric_types = ['global_importance', 'feature_selection_frequency', 'phi_importance', 'coverage_weighted_importance']
        
        for i, metric_type in enumerate(metric_types):
            plt.subplot(2, 2, i+1)
            
            top_strat = f"{metric_type}_top{self.args.top_k if hasattr(self.args, 'top_k') else 5}"
            bottom_strat = f"{metric_type}_bottom{self.args.top_k if hasattr(self.args, 'top_k') else 5}"
            
            top_df = results_df[results_df['Strategy'] == top_strat]
            bottom_df = results_df[results_df['Strategy'] == bottom_strat]
            
            if not top_df.empty and not bottom_df.empty:
                top_values = top_df[metrics].values[0]
                bottom_values = bottom_df[metrics].values[0]
                
                df = pd.DataFrame({
                    'Metric': metrics,
                    'Top': top_values,
                    'Bottom': bottom_values
                })
                
                df.plot(x='Metric', y=['Top', 'Bottom'], kind='bar', ax=plt.gca())
                plt.title(f'{metric_type} - Top vs Bottom')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f"{self.args.data}_top_vs_bottom.png"))
        plt.close()