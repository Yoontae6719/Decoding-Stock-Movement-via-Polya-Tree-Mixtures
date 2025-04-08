# ./exp/exp_main.py

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
from utils.xai_standalone import TreeXAI
from collections import defaultdict

from utils.tools import EarlyStopping, adjust_learning_rate, TemperatureScheduler
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score
import shutil
import optuna
import gc

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
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
        vali_data, vali_loader   = self._get_data(flag='val')
        test_data, test_loader   = self._get_data(flag='test')        
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        
        outputs_path = f'./final_{self.args.data}/{setting}/'

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
            
            # validation
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
        
        # Run XAI analysis after training is complete
        # Use the temperature that gave the best validation performance
        print(f"Running XAI analysis with best temperature: {best_temp}")
        xai_results = self.xai(setting=setting, current_temp=best_temp)
        
        # Save XAI summary to the outputs directory
        try:
            xai_dir = os.path.join(outputs_path, 'xai_results')
            if not os.path.exists(xai_dir):
                os.makedirs(xai_dir)
            
            # 1. Summary metrics
            xai_summary = {
                'accuracy': xai_results['metrics']['accuracy'],
                'mcc': xai_results['metrics']['mcc'],
                'f1_score': xai_results['metrics']['f1_score']
            }
            
            with open(os.path.join(xai_dir, 'summary.json'), 'w') as f:
                json.dump(xai_summary, f, indent=2)
            
            # Create a DataFrame for summary metrics
            summary_df = pd.DataFrame([xai_summary])
            summary_df.to_csv(os.path.join(xai_dir, 'summary.csv'), index=False)
            
            # 2. Node statistics as CSV
            node_stats_list = []
            for node_name, stats in xai_results['node_statistics'].items():
                node_data = {
                    'node_name': node_name,
                    'type': stats['type'],
                    'depth': int(stats['depth']),
                    'coverage': float(stats['coverage']),
                    'samples': int(stats['samples'])
                }
                
                # Add leaf-specific metrics
                if stats['type'] == 'leaf':
                    node_data['accuracy'] = float(stats.get('accuracy', 0))
                    if 'class_distribution' in stats:
                        # Convert class distribution to columns
                        for i, val in enumerate(stats['class_distribution']):
                            node_data[f'class_{i}'] = float(val)
                
                node_stats_list.append(node_data)
            
            # Convert to DataFrame and save
            node_stats_df = pd.DataFrame(node_stats_list)
            node_stats_df.to_csv(os.path.join(xai_dir, 'node_statistics.csv'), index=False)
            
            # 3. Feature importance per node as CSV
            # Get feature names
            feature_names = [f"Feature_{i}" for i in range(self.args.dim_input)]
            if hasattr(test_data, 'feature_names'):
                feature_names = test_data.feature_names
            
            # 3a. Save top features per node
            for node_name, importance in xai_results['feature_importance'].items():
                imp_values = importance['importance']
                # Create a dataframe with all features
                node_imp_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': imp_values
                })
                node_imp_df = node_imp_df.sort_values('importance', ascending=False)
                
                # Save to CSV
                node_path = os.path.join(xai_dir, 'nodes')
                if not os.path.exists(node_path):
                    os.makedirs(node_path)
                
                node_imp_df.to_csv(os.path.join(node_path, f'{node_name}_features.csv'), index=False)
            
            # 3b. Save metadata about each node
            node_meta_list = []
            for node_name, importance in xai_results['feature_importance'].items():
                node_meta_list.append({
                    'node_name': node_name,
                    'node_type': importance['node_type'],
                    'depth': int(importance['depth'])
                })
            
            node_meta_df = pd.DataFrame(node_meta_list)
            node_meta_df.to_csv(os.path.join(xai_dir, 'node_metadata.csv'), index=False)
            
            # 4. Global feature importance as CSV
            global_imp_df = pd.DataFrame({
                'feature': feature_names,
                'importance': xai_results['global_importance']
            })
            global_imp_df = global_imp_df.sort_values('importance', ascending=False)
            global_imp_df.to_csv(os.path.join(xai_dir, 'global_importance.csv'), index=False)
            
            # 5. Leaf paths as CSV
            leaf_paths_list = []
            for node_name, stats in xai_results['node_statistics'].items():
                if stats['type'] == 'leaf':
                    # Extract path from node name (e.g., "root_L_R")
                    path_parts = node_name.split('_')
                    path_str = " -> ".join(path_parts)
                    
                    leaf_paths_list.append({
                        'leaf_name': node_name,
                        'path': path_str,
                        'coverage': float(stats['coverage']),
                        'accuracy': float(stats.get('accuracy', 0)),
                        'samples': int(stats['samples'])
                    })
            
            leaf_paths_df = pd.DataFrame(leaf_paths_list)
            leaf_paths_df.to_csv(os.path.join(xai_dir, 'leaf_paths.csv'), index=False)
            
            # 6. Additional feature importance metrics from XAI results
            if 'additional_importances' in xai_results:
                for metric_name, values in xai_results['additional_importances'].items():
                    metric_df = pd.DataFrame({
                        'feature': feature_names,
                        metric_name: values
                    })
                    metric_df = metric_df.sort_values(metric_name, ascending=False)
                    metric_df.to_csv(os.path.join(xai_dir, f'{metric_name}.csv'), index=False)
            
            print(f"XAI analysis complete. Detailed results saved to {xai_dir}")
        except Exception as e:
            print(f"Error saving XAI results: {e}")
            import traceback
            traceback.print_exc()


    def evaluate(self, data_loader, current_temp):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        preds_list = []
        labels_list = []

        #test_temp = 0.2
        with torch.no_grad():
            for x_batch, y_batch, x_close in data_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device).squeeze(-1)

                probs = self.model(x_batch,temperature_fs=current_temp, temperature_gate=0.5)

                #probs = self.model.forward_hard(x_batch)
                nll = -torch.log(probs[torch.arange(x_batch.size(0)), y_batch] + 1e-8).sum().item()
                #kl_loss = self.model.get_kl_loss().sum().item()

                preds = torch.argmax(probs, dim=1)
                correct = (preds == y_batch).sum().item()
                
                total_loss += nll #+ self.args.lambda_KL * kl_loss
                total_correct += correct
                total_samples += x_batch.size(0)

                preds_list.extend(preds.cpu().numpy())
                labels_list.extend(y_batch.cpu().numpy())

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        mcc = matthews_corrcoef(labels_list, preds_list)
        f1_macro = f1_score(labels_list, preds_list, average='macro')

        return avg_loss, avg_acc, mcc, f1_macro
        

    def predict(self, x, temp):
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            logits = self.model(x, temperature_fs=temp, temperature_gate = temp)
            preds = torch.argmax(logits, dim=1)
        return preds.cpu()
    
    def xai(self, setting, current_temp):
        # Get test data
        test_data, test_loader = self._get_data(flag='test')
        
        # Create output directory
        outputs_path = f'./xai_results_{self.args.data}/{setting}/'
        if not os.path.exists(outputs_path):
            os.makedirs(outputs_path)
        
        # Move model to evaluation mode
        self.model.eval()
        
        # Prepare containers for predictions and labels
        all_preds = []
        all_labels = []
        leaf_nodes = {}  # To track leaf node usage
        
        # Initialize node statistics
        node_statistics = {}
        feature_importance = {}
        
        # 1. First pass: collect predictions and analyze model performance
        with torch.no_grad():
            for x_batch, y_batch, x_close in test_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device).squeeze(-1)
                
                # Forward pass - using same settings as evaluate()
                probs = self.model(x_batch, temperature_fs=current_temp, temperature_gate=0.5)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        acc = (np.array(all_preds) == np.array(all_labels)).mean()
        mcc = matthews_corrcoef(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        print(f"Overall metrics - Accuracy: {acc:.4f}, MCC: {mcc:.4f}, F1: {f1:.4f}")
        with open(os.path.join(outputs_path, "overall_metrics.txt"), 'w') as f:
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"MCC: {mcc:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(f"\nConfusion Matrix:\n{conf_matrix}\n")
        
        # 2. Extract feature importance from tree model
        node_coverage = defaultdict(int)
        node_classes = defaultdict(lambda: np.zeros(self.args.num_classes))
        node_correct = defaultdict(int)
        node_samples = defaultdict(int)
        total_samples = 0
        
        # Function to recursively analyze tree nodes
        def analyze_node(node, node_name, threshold_fs=0.5):
            if node.is_leaf:
                if node.use_feature_selection and node.fs_module is not None: # Get feature importance from leaf node
                    phi = torch.sigmoid(node.fs_module.logit_phi).detach().cpu().numpy()
                    
                    
                    fc1_weights = node.local_expert.fc1.weight.detach().cpu().numpy() # Get weights from local expert
                    alpha = np.mean(np.abs(fc1_weights), axis=0)
                    
                   
                    feat_importance = phi * alpha # Calculate feature importance
                    
                    feature_importance[node_name] = {
                        'importance': feat_importance,
                        'phi': phi,
                        'alpha': alpha,
                        'selected_features': np.where(phi > threshold_fs)[0],
                        'node_type': 'leaf',
                        'depth': node.depth
                    }
                else:
                    feat_importance = np.ones(self.args.dim_input)
                    if hasattr(node, 'local_expert') and hasattr(node.local_expert, 'fc1'):
                        fc1_weights = node.local_expert.fc1.weight.detach().cpu().numpy()
                        alpha = np.mean(np.abs(fc1_weights), axis=0)
                        feat_importance = alpha
                    
                    feature_importance[node_name] = {
                        'importance': feat_importance,
                        'node_type': 'leaf',
                        'depth': node.depth
                    }
                
                node_statistics[node_name] = {
                    'type': 'leaf',
                    'depth': node.depth,
                    'coverage': 0,
                    'samples': 0,
                    'accuracy': 0,
                    'mcc': 0
                }
            else:
                if node.use_feature_selection and node.fs_module is not None: # Internal node
                    phi = torch.sigmoid(node.fs_module.logit_phi).detach().cpu().numpy()
                    
                   
                    if node.gating is not None:  # Get gating weights
                        gating_weights = node.gating[0].weight.detach().cpu().numpy()
                        alpha = np.mean(np.abs(gating_weights), axis=0)
                    else:
                        alpha = np.abs(node.w.detach().cpu().numpy())
                    
                    feat_importance = phi * alpha
                    
                    feature_importance[node_name] = {
                        'importance': feat_importance,
                        'phi': phi,
                        'alpha': alpha,
                        'selected_features': np.where(phi > threshold_fs)[0],
                        'node_type': 'internal',
                        'depth': node.depth
                    }
                else:
                    feat_importance = np.ones(self.args.dim_input)
                    if node.gating is not None:
                        gating_weights = node.gating[0].weight.detach().cpu().numpy()
                        alpha = np.mean(np.abs(gating_weights), axis=0)
                        feat_importance = alpha
                    elif hasattr(node, 'w'):
                        alpha = np.abs(node.w.detach().cpu().numpy())
                        feat_importance = alpha
                    
                    feature_importance[node_name] = {
                        'importance': feat_importance,
                        'node_type': 'internal',
                        'depth': node.depth
                    }
                
                node_statistics[node_name] = {
                    'type': 'internal',
                    'depth': node.depth,
                    'coverage': 0,
                    'samples': 0
                }
                
                
                analyze_node(node.left_child, f"{node_name}_L", threshold_fs)  # Recursively analyze children
                analyze_node(node.right_child, f"{node_name}_R", threshold_fs) # Recursively analyze children
        
        analyze_node(self.model.root, "root")
        
        def trace_samples(x_batch, y_batch):
            """Trace the path of each sample through the tree and update node statistics"""
            nonlocal total_samples
            batch_size = x_batch.size(0)
            total_samples += batch_size
            
            
            for i in range(batch_size): # For each sample in batch
                x = x_batch[i:i+1]
                label = y_batch[i].item()
                
                current_node = self.model.root # Trace path through tree
                node_path = ["root"]
                node_samples["root"] += 1
                node_classes["root"][label] += 1
                
                while not current_node.is_leaf:
                    if current_node.use_feature_selection: # Apply feature selection if needed
                        phi = torch.sigmoid(current_node.fs_module.logit_phi)
                        gamma = (phi > 0.5).float()
                        x_masked = x * gamma.unsqueeze(0)
                    else:
                        x_masked = x
                    
                    if current_node.gating is not None: # Determine which child to go to
                        gate_logit = current_node.gating(x_masked).squeeze(-1)
                    else:
                        gate_logit = (x_masked * current_node.w).sum(dim=1) + current_node.b
                    
                    p_left = torch.sigmoid(gate_logit / current_temp)  # Use current_temp
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
                
                with torch.no_grad():
                    pred_probs = self.model(x, temperature_fs=current_temp, temperature_gate=0.5)
                    pred = torch.argmax(pred_probs, dim=1).item()
                
                if pred == label:
                    node_correct[leaf_name] += 1
        
        self.model.eval()
        with torch.no_grad():
            for x_batch, y_batch, _ in test_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device).squeeze(-1)
                trace_samples(x_batch, y_batch)
        
        for node_name in node_statistics:
            coverage = node_samples[node_name] / total_samples if total_samples > 0 else 0
            node_statistics[node_name]['coverage'] = coverage
            node_statistics[node_name]['samples'] = node_samples[node_name]
            
            if node_statistics[node_name]['type'] == 'leaf':
                if node_samples[node_name] > 0:
                    accuracy = node_correct[node_name] / node_samples[node_name]
                    node_statistics[node_name]['accuracy'] = accuracy
                    class_dist = node_classes[node_name]
                    node_statistics[node_name]['class_distribution'] = class_dist
        
        # 4. Calculate global feature importance by weighting node importance
        global_importance = np.zeros(self.args.dim_input)
        
        for node_name, importance in feature_importance.items():
            node_stats = node_statistics[node_name]
            node_coverage = node_stats['coverage']
            
            if node_stats['type'] == 'leaf':
                node_accuracy = node_stats.get('accuracy', 0.5)
                node_weight = node_coverage * node_accuracy
            else:
                node_weight = node_coverage
            
            global_importance += node_weight * importance['importance']
        
        if np.sum(global_importance) > 0:
            global_importance = global_importance / np.sum(global_importance)
        
        # 5. Calculate additional feature importance metrics (I will revised this part)
        
        feature_selection_freq = np.zeros(self.args.dim_input)  # Feature selection frequency (how often each feature is selected across nodes)
        selection_weights = np.zeros(self.args.dim_input)
        phi_importance = np.zeros(self.args.dim_input)          # Phi importance (just the average phi values across nodes)
        alpha_importance = np.zeros(self.args.dim_input)        # Alpha importance (just the average alpha values across nodes)
        leaf_importance = np.zeros(self.args.dim_input)         # Leaf-only importance (considering only leaf nodes)
        
        total_nodes = 0
        total_leaves = 0
        
        for node_name, importance in feature_importance.items():
            total_nodes += 1
            node_stats = node_statistics[node_name]
            
            if 'phi' in importance:
                phi = importance['phi']
                phi_importance += phi
            elif 'importance' in importance:
                imp = importance['importance']
                phi_importance += imp
            
            if 'alpha' in importance:
                alpha = importance['alpha']
                alpha_importance += alpha
            

            selected_features = importance.get('selected_features', [])
            if len(selected_features) > 0:
                feature_selection_freq[selected_features] += 1
                selection_weights[selected_features] += node_stats['coverage']
            
            if node_stats['type'] == 'leaf':
                total_leaves += 1
                leaf_importance += node_stats['coverage'] * importance['importance']
        
        if total_nodes > 0:
            phi_importance /= total_nodes
            alpha_importance /= total_nodes
            feature_selection_freq /= total_nodes
        
        if np.sum(leaf_importance) > 0:
            leaf_importance /= np.sum(leaf_importance)
        
        additional_importances = {
            'phi_importance': phi_importance,
            'alpha_importance': alpha_importance,
            'feature_selection_frequency': feature_selection_freq,
            'selection_weights': selection_weights,
            'leaf_only_importance': leaf_importance
        }
        
        coverage_weighted_importance = np.zeros(self.args.dim_input) # Add weighted importance versions
        for node_name, importance in feature_importance.items():
            node_stats = node_statistics[node_name]
            coverage_weighted_importance += node_stats['coverage'] * importance['importance']
        
        if np.sum(coverage_weighted_importance) > 0:
            coverage_weighted_importance /= np.sum(coverage_weighted_importance)
        
        additional_importances['coverage_weighted_importance'] = coverage_weighted_importance
        
        internal_importance = np.zeros(self.args.dim_input)
        internal_nodes = 0
        
        for node_name, importance in feature_importance.items():
            node_stats = node_statistics[node_name]
            if node_stats['type'] == 'internal':
                internal_nodes += 1
                internal_importance += importance['importance']
        
        if internal_nodes > 0:
            internal_importance /= internal_nodes
        
        additional_importances['internal_nodes_importance'] = internal_importance
        
        # 5. Save 
        feature_names = [f"Feature_{i}" for i in range(self.args.dim_input)]
        if hasattr(test_data, 'feature_names'):
            feature_names = test_data.feature_names
        
        with open(os.path.join(outputs_path, "node_statistics.txt"), 'w') as f:
            for name, stats in node_statistics.items():
                if stats['type'] == 'internal':
                    f.write(f"[Node] {name} | depth={stats['depth']} | coverage={stats['coverage']:.4f} | "
                        f"samples={stats['samples']}\n")
                else:  # leaf
                    f.write(f"[Leaf] {name} | depth={stats['depth']} | coverage={stats['coverage']:.4f} | "
                        f"samples={stats['samples']} | "
                        f"acc={stats.get('accuracy', 0):.4f}\n")

        with open(os.path.join(outputs_path, "global_feature_importance.txt"), 'w') as f:
            f.write("Global Feature Importance:\n")            
            top_indices = np.argsort(global_importance)[-30:]
            for idx in reversed(top_indices):
                f.write(f"{feature_names[idx]}: {global_importance[idx]:.4f}\n")
        
        global_imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': global_importance
        })
        global_imp_df = global_imp_df.sort_values('importance', ascending=False)
        global_imp_df.to_csv(os.path.join(outputs_path, "global_feature_importance.csv"), index=False)        
        print("Saving additional feature importance metrics...")
        
        freq_df = pd.DataFrame({
            'feature': feature_names,
            'selection_frequency': feature_selection_freq,
            'selection_weight': selection_weights
        })
        freq_df = freq_df.sort_values('selection_frequency', ascending=False)
        freq_df.to_csv(os.path.join(outputs_path, "feature_selection_frequency.csv"), index=False)
        
        phi_df = pd.DataFrame({
            'feature': feature_names,
            'phi_importance': phi_importance
        })
        phi_df = phi_df.sort_values('phi_importance', ascending=False)
        phi_df.to_csv(os.path.join(outputs_path, "phi_importance.csv"), index=False)
        
        alpha_df = pd.DataFrame({
            'feature': feature_names,
            'alpha_importance': alpha_importance
        })
        alpha_df = alpha_df.sort_values('alpha_importance', ascending=False)
        alpha_df.to_csv(os.path.join(outputs_path, "alpha_importance.csv"), index=False)
        
        leaf_df = pd.DataFrame({
            'feature': feature_names,
            'leaf_importance': leaf_importance
        })
        leaf_df = leaf_df.sort_values('leaf_importance', ascending=False)
        leaf_df.to_csv(os.path.join(outputs_path, "leaf_only_importance.csv"), index=False)
        
        cov_df = pd.DataFrame({
            'feature': feature_names,
            'coverage_weighted_importance': coverage_weighted_importance
        })
        cov_df = cov_df.sort_values('coverage_weighted_importance', ascending=False)
        cov_df.to_csv(os.path.join(outputs_path, "coverage_weighted_importance.csv"), index=False)
        
        internal_df = pd.DataFrame({
            'feature': feature_names,
            'internal_nodes_importance': internal_importance
        })
        internal_df = internal_df.sort_values('internal_nodes_importance', ascending=False)
        internal_df.to_csv(os.path.join(outputs_path, "internal_nodes_importance.csv"), index=False)
        
        print("\nTop 10 Global Features:")
        for i, row in global_imp_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return {
            'metrics': {'accuracy': acc, 'mcc': mcc, 'f1_score': f1},
            'node_statistics': node_statistics,
            'feature_importance': feature_importance,
            'global_importance': global_importance,
            'additional_importances': additional_importances
        }
