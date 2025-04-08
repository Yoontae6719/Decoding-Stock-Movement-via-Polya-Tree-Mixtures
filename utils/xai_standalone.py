# .utils/xai_standalone.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, confusion_matrix
import os
import sys
import json
from collections import defaultdict

# Import project modules
from models import treeMoE
from data_provider.data_factory import data_provider
from utils.tools import dotdict

class TreeXAI:
    def __init__(self, model, device, feature_names=None, class_names=None):
        self.model = model
        self.device = device
        self.feature_names = feature_names if feature_names is not None else [f"Feature_{i}" for i in range(model.dim_input)]
        self.class_names = class_names if class_names is not None else [f"Class_{i}" for i in range(model.num_classes)]
        
        # Initialize tree statistics
        self.node_statistics = {}
        self.feature_importance = {}
        self.leaf_importance = {}
        self.node_coverage = {}
        self.node_paths = {}
        self.global_feature_importance = None
        
    def analyze(self, data_loader, threshold_fs=0.5, top_k=30, output_dir="./xai_results", 
                leaf_importance_method='product', alpha_coverage=1.0, alpha_acc=1.0, alpha_mcc=0.5):

        os.makedirs(output_dir, exist_ok=True)
        
        all_preds, all_labels, all_probs = self._collect_predictions(data_loader)
        overall_metrics = self._calculate_overall_metrics(all_preds, all_labels)
        self._analyze_tree_structure(self.model.root, "root", threshold_fs)
        self._analyze_node_statistics(data_loader, threshold_fs)
        self._calculate_node_feature_importance(threshold_fs)        
        self._calculate_leaf_importance(method=leaf_importance_method, 
                                      alpha_coverage=alpha_coverage, 
                                      alpha_acc=alpha_acc, 
                                      alpha_mcc=alpha_mcc)
        
        self._calculate_global_feature_importance()        
        self._save_results(output_dir, overall_metrics, top_k)
        
        return {
            'metrics': overall_metrics,
            'global_feature_importance': self.global_feature_importance,
            'leaf_importance': self.leaf_importance
        }
    
    def _collect_predictions(self, data_loader):
        all_preds = []
        all_labels = []
        all_probs = []
        
        self.model.eval()
        with torch.no_grad():
            for x_batch, y_batch, _ in data_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device).squeeze(-1)
                
                probs = self.model.forward_hard(x_batch, threshold_fs=0.5)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_labels), np.array(all_probs)
    
    def _calculate_overall_metrics(self, predictions, true_labels):
        acc = accuracy_score(true_labels, predictions)
        mcc = matthews_corrcoef(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        return {
            'accuracy': acc,
            'mcc': mcc,
            'f1_score': f1,
            'confusion_matrix': conf_matrix
        }
    
    def _analyze_tree_structure(self, node, node_name, threshold_fs):
        if node.is_leaf:
            self._analyze_leaf_node(node, node_name, threshold_fs)
        else:
            self._analyze_internal_node(node, node_name, threshold_fs)
            
            self._analyze_tree_structure(node.left_child, f"{node_name}_L", threshold_fs)
            self._analyze_tree_structure(node.right_child, f"{node_name}_R", threshold_fs)
    
    def _analyze_leaf_node(self, node, node_name, threshold_fs):
        if node.use_feature_selection and node.fs_module is not None:
            phi = torch.sigmoid(node.fs_module.logit_phi).detach().cpu().numpy()
            gamma = (phi > threshold_fs).astype(float)
            fc1_weights = node.local_expert.fc1.weight.detach().cpu().numpy()            
            alpha = np.mean(np.abs(fc1_weights), axis=0)
            
            feature_importance = phi * alpha
            
            self.feature_importance[node_name] = {
                'phi': phi,
                'gamma': gamma,
                'alpha': alpha,
                'selected_features': np.where(gamma > 0)[0],
                'feature_importance': feature_importance,
                'node_type': 'leaf',
                'depth': node.depth
            }
        else:
            alpha = np.ones(self.model.dim_input)
            
            if hasattr(node, 'local_expert') and hasattr(node.local_expert, 'fc1'):
                fc1_weights = node.local_expert.fc1.weight.detach().cpu().numpy()
                alpha = np.mean(np.abs(fc1_weights), axis=0)
                
            self.feature_importance[node_name] = {
                'phi': np.ones(self.model.dim_input),
                'gamma': np.ones(self.model.dim_input),
                'alpha': alpha,
                'selected_features': np.arange(self.model.dim_input),
                'feature_importance': alpha,  # Without phi, just use alpha
                'node_type': 'leaf',
                'depth': node.depth
            }
            
        self.node_statistics[node_name] = {
            'type': 'leaf',
            'depth': node.depth,
            'coverage': 0,  # Will be updated during data analysis
            'class_distribution': np.zeros(self.model.num_classes),  # Will be updated
            'accuracy': 0,  # Will be updated
            'mcc': 0  # Will be updated
        }
    
    def _analyze_internal_node(self, node, node_name, threshold_fs):
        if node.use_feature_selection and node.fs_module is not None:
            phi = torch.sigmoid(node.fs_module.logit_phi).detach().cpu().numpy()
            gamma = (phi > threshold_fs).astype(float)
        else:
            phi = np.ones(self.model.dim_input)
            gamma = np.ones(self.model.dim_input)
            
        if node.gating is not None:
            gating_weights = node.gating[0].weight.detach().cpu().numpy()
            alpha = np.mean(np.abs(gating_weights), axis=0)
        else:
            alpha = np.abs(node.w.detach().cpu().numpy())
            
        feature_importance = phi * alpha
            
        self.feature_importance[node_name] = {
            'phi': phi,
            'gamma': gamma,
            'alpha': alpha,
            'selected_features': np.where(gamma > 0)[0],
            'feature_importance': feature_importance,
            'node_type': 'internal',
            'depth': node.depth
        }
        
        self.node_statistics[node_name] = {
            'type': 'internal',
            'depth': node.depth,
            'coverage': 0,  # Will be updated during data analysis
            'left_prob': 0,  # Will be updated
            'right_prob': 0,  # Will be updated
            'class_distribution': np.zeros(self.model.num_classes)  # Will be updated
        }
    
    def _analyze_node_statistics(self, data_loader, threshold_fs):
        node_samples = defaultdict(int)
        node_classes = defaultdict(lambda: np.zeros(self.model.num_classes))
        node_correct = defaultdict(int)
        node_predictions = defaultdict(list)
        node_true_labels = defaultdict(list)
        
        total_samples = 0
        
        self.model.eval()
        with torch.no_grad():
            for x_batch, y_batch, _ in data_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device).squeeze(-1)
                batch_size = x_batch.size(0)
                total_samples += batch_size
                
                paths = self._trace_sample_paths(self.model.root, x_batch, "root", threshold_fs)
                
                for i, (path, leaf) in enumerate(paths):
                    label = y_batch[i].item()
                    
                    for node_name in path:
                        node_samples[node_name] += 1
                        node_classes[node_name][label] += 1
                    
                    leaf_name = path[-1]
                    node_predictions[leaf_name].append(
                        torch.argmax(self.model.forward_hard(x_batch[i:i+1], threshold_fs)).item()
                    )
                    node_true_labels[leaf_name].append(label)
                    if node_predictions[leaf_name][-1] == label:
                        node_correct[leaf_name] += 1
        
        for node_name in self.node_statistics:
            coverage = node_samples[node_name] / total_samples if total_samples > 0 else 0
            self.node_statistics[node_name]['coverage'] = coverage
            self.node_statistics[node_name]['sample_count'] = node_samples[node_name]
            self.node_statistics[node_name]['class_distribution'] = node_classes[node_name]
            
            if self.node_statistics[node_name]['type'] == 'leaf':
                if node_samples[node_name] > 0:
                    accuracy = node_correct[node_name] / node_samples[node_name]
                    
                    if len(set(node_true_labels[node_name])) > 1 and len(node_true_labels[node_name]) > 1:
                        mcc = matthews_corrcoef(node_true_labels[node_name], node_predictions[node_name])
                    else:
                        mcc = 0  # Default if we can't calculate
                    
                    self.node_statistics[node_name]['accuracy'] = accuracy
                    self.node_statistics[node_name]['mcc'] = mcc
                    self.node_statistics[node_name]['predictions'] = node_predictions[node_name]
                    self.node_statistics[node_name]['true_labels'] = node_true_labels[node_name]
        
        self.node_coverage = {node: stats['coverage'] for node, stats in self.node_statistics.items()}
    
    def _trace_sample_paths(self, node, x, node_name, threshold_fs, path=None):
        if path is None:
            paths = [([node_name], None) for _ in range(x.size(0))]
        else:
            paths = path
        
        if node.is_leaf:
            updated_paths = []
            for path_nodes, _ in paths:
                updated_paths.append((path_nodes, node_name))
            return updated_paths
        
        if node.use_feature_selection:
            phi = torch.sigmoid(node.fs_module.logit_phi)
            gamma = (phi > threshold_fs).float()
            x_masked = x * gamma.unsqueeze(0)
        else:
            x_masked = x
        
        if node.gating is not None:
            gate_logit = node.gating(x_masked).squeeze(-1)
        else:
            gate_logit = (x_masked * node.w).sum(dim=1) + node.b
        
        p_left = torch.sigmoid(gate_logit)
        go_left = (p_left > 0.5).cpu().numpy()
        
        left_indices = np.where(go_left)[0]
        right_indices = np.where(~go_left)[0]
        
        left_paths = []
        right_paths = []
        
        for i, (path_nodes, _) in enumerate(paths):
            if i in left_indices:
                left_paths.append((path_nodes + [f"{node_name}_L"], None))
            elif i in right_indices:
                right_paths.append((path_nodes + [f"{node_name}_R"], None))
        
        left_results = []
        right_results = []
        
        if left_paths:
            left_x = x_masked[left_indices]
            left_results = self._trace_sample_paths(node.left_child, left_x, f"{node_name}_L", threshold_fs, left_paths)
        
        if right_paths:
            right_x = x_masked[right_indices]
            right_results = self._trace_sample_paths(node.right_child, right_x, f"{node_name}_R", threshold_fs, right_paths)
        
        all_results = [None] * len(paths)
        left_idx = 0
        right_idx = 0
        
        for i in range(len(paths)):
            if i in left_indices:
                all_results[i] = left_results[left_idx]
                left_idx += 1
            elif i in right_indices:
                all_results[i] = right_results[right_idx]
                right_idx += 1
        
        return all_results
    
    def _calculate_node_feature_importance(self, threshold_fs):

        for node_name, importance in self.feature_importance.items():
            fi = importance['feature_importance']
            if np.sum(fi) > 0:
                importance['normalized_importance'] = fi / np.sum(fi)
            else:
                importance['normalized_importance'] = np.ones_like(fi) / len(fi)
    
    def _calculate_leaf_importance(self, method='product', alpha_coverage=1.0, alpha_acc=1.0, alpha_mcc=0.5):

        leaf_nodes = {name: stats for name, stats in self.node_statistics.items() 
                      if stats['type'] == 'leaf'}
        
        for name, stats in leaf_nodes.items():
            coverage_ratio = stats['coverage']
            
            accuracy = stats['accuracy']
            mcc = stats['mcc']
            
            if method == 'product':
                importance = (coverage_ratio ** alpha_coverage) * (accuracy ** alpha_acc) * ((1 + mcc) ** alpha_mcc)
            else:  # 'sum'
                importance = (alpha_coverage * coverage_ratio) + (alpha_acc * accuracy) + (alpha_mcc * (1 + mcc) / 2)
            
            self.leaf_importance[name] = {
                'coverage': coverage_ratio,
                'accuracy': accuracy,
                'mcc': mcc,
                'importance': importance,
                'depth': stats['depth'],
                'sample_count': stats['sample_count']
            }
            
        total_importance = sum(info['importance'] for info in self.leaf_importance.values())
        if total_importance > 0:
            for name in self.leaf_importance:
                self.leaf_importance[name]['normalized_importance'] = self.leaf_importance[name]['importance'] / total_importance
    
    def _calculate_global_feature_importance(self):
        global_importance = np.zeros(self.model.dim_input)
        
        for node_name, importance in self.feature_importance.items():
            node_stats = self.node_statistics[node_name]
            node_coverage = node_stats['coverage']
            
            if node_stats['type'] == 'leaf':
                node_accuracy = node_stats['accuracy']
                node_mcc = node_stats['mcc']
                node_weight = node_coverage * (node_accuracy * (1 + node_mcc) / 2)
            else:
                node_weight = node_coverage
            
            node_fi = importance['feature_importance']
            
            global_importance += node_weight * node_fi
        
        if np.sum(global_importance) > 0:
            global_importance = global_importance / np.sum(global_importance)
        
        self.global_feature_importance = global_importance
    
    def _save_results(self, output_dir, overall_metrics, top_k=10):
        # 1. Overall metrics
        with open(os.path.join(output_dir, "overall_metrics.txt"), 'w') as f:
            f.write(f"Accuracy: {overall_metrics['accuracy']:.4f}\n")
            f.write(f"MCC: {overall_metrics['mcc']:.4f}\n")
            f.write(f"F1 Score: {overall_metrics['f1_score']:.4f}\n")
            f.write(f"\nConfusion Matrix:\n{overall_metrics['confusion_matrix']}\n")
        
        # 2. Node statistics
        with open(os.path.join(output_dir, "node_statistics.txt"), 'w') as f:
            for name, stats in self.node_statistics.items():
                if stats['type'] == 'internal':
                    f.write(f"[Node] {name} | depth={stats['depth']} | coverage={stats['coverage']:.4f} | "
                           f"samples={stats['sample_count']} | "
                           f"class_dist={stats['class_distribution']}\n")
                else:  # leaf
                    f.write(f"[Leaf] {name} | depth={stats['depth']} | coverage={stats['coverage']:.4f} | "
                           f"samples={stats['sample_count']} | "
                           f"acc={stats['accuracy']:.4f} | mcc={stats['mcc']:.4f} | "
                           f"class_dist={stats['class_distribution']}\n")
        
        # 3. Feature importance for each leaf
        with open(os.path.join(output_dir, "feature_importance.txt"), 'w') as f:
            for node_name, importance in self.feature_importance.items():
                if self.node_statistics[node_name]['type'] == 'leaf':
                    f.write(f"Leaf: {node_name} (depth={importance['depth']}, "
                           f"coverage={self.node_statistics[node_name]['coverage']:.4f})\n")
                    
                    # Get feature importance
                    imp_scores = importance['feature_importance']
                    
                    # Get top K important features
                    top_indices = np.argsort(imp_scores)[-top_k:]
                    top_features = [(self.feature_names[i], imp_scores[i]) for i in top_indices]
                    
                    # Get bottom K unimportant features
                    bottom_indices = np.argsort(imp_scores)[:top_k]
                    bottom_features = [(self.feature_names[i], imp_scores[i]) for i in bottom_indices]
                    
                    f.write("Top Important Features:\n")
                    for feature, score in reversed(top_features):
                        f.write(f"  {feature}: {score:.4f}\n")
                    
                    f.write("Least Important Features:\n")
                    for feature, score in bottom_features:
                        f.write(f"  {feature}: {score:.4f}\n")
                    
                    f.write("\n")
        
        # 4. Leaf importance
        with open(os.path.join(output_dir, "leaf_importance.txt"), 'w') as f:
            # Sort by importance
            sorted_leaves = sorted(self.leaf_importance.items(), 
                                 key=lambda x: x[1]['importance'], 
                                 reverse=True)
            
            f.write("Leaf Importance (sorted):\n")
            for name, info in sorted_leaves:
                f.write(f"{name}: depth={info['depth']} | coverage={info['coverage']:.4f} | "
                       f"samples={info['sample_count']} | "
                       f"accuracy={info['accuracy']:.4f} | mcc={info['mcc']:.4f} | "
                       f"importance={info['importance']:.4f}\n")
        
        # 5. Global Feature Importance
        with open(os.path.join(output_dir, "global_feature_importance.txt"), 'w') as f:
            f.write("Global Feature Importance:\n")
            
            top_indices = np.argsort(self.global_feature_importance)[-top_k:]
            for idx in reversed(top_indices):
                f.write(f"{self.feature_names[idx]}: {self.global_feature_importance[idx]:.4f}\n")
                
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.global_feature_importance
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df.to_csv(os.path.join(output_dir, "global_feature_importance.csv"), index=False)


def load_model_from_checkpoint(checkpoint_path, args, device):

    print(f"Loading model from {checkpoint_path}")
    

    model = treeMoE.Model(args)
    
    try:
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint file not found: {checkpoint_path}")
            return None
        
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def extract_feature_names(data_set, dim_input):
    if hasattr(data_set, 'feature_names') and data_set.feature_names is not None:
        return data_set.feature_names
    
    categories = [
        "Price", "Volume", "Momentum", "Technical", "Volatility", 
        "Trend", "Pattern", "Oscillator", "Liquidity", "Sentiment"
    ]
    
    feature_names = []
    for i in range(dim_input):
        category = categories[i % len(categories)]
        feature_names.append(f"{category}_{i//len(categories) + 1}")
    
    return feature_names


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='TreeMoE XAI Analysis')
    
    parser.add_argument('--checkpoint', type=str, required=True, help='path to model checkpoint')
    parser.add_argument('--model_id', type=str, required=True, help='model id')
    parser.add_argument('--data', type=str, required=True, help='dataset name')
    parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
    parser.add_argument('--data_path', type=str, required=True, help='data file path')
    
    parser.add_argument('--dim_input', type=int, required=True, help='input dimension')
    parser.add_argument('--num_classes', type=int, default=2, help='output dimension')
    parser.add_argument('--max_depth', type=int, required=True, help='tree max depth')
    parser.add_argument('--use_feature_selection', action='store_true', default=True, help='use feature selection')
    parser.add_argument('--use_leaf_feature_selector_only', type=int, default=1, help='use leaf feature selector only')
    parser.add_argument('--use_gating_mlp', type=int, default=0, help='use gating mlp')
    parser.add_argument('--gating_mlp_hidden', type=int, default=32, help='gating mlp hidden dim')
    parser.add_argument('--alpha_fs', type=float, default=1.0, help='Beta-Bernoulli alpha')
    parser.add_argument('--beta_fs', type=float, default=1.0, help='Beta-Bernoulli beta')
    parser.add_argument('--hidden_dim_expert', type=int, default=32, help='hidden dim for expert')
    
    parser.add_argument('--output_dir', type=str, default=None, help='output directory')
    parser.add_argument('--top_k', type=int, default=30, help='number of top features to report')
    parser.add_argument('--threshold_fs', type=float, default=0.5, help='threshold for feature selection')
    
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--scale', type=bool, default=True, help='scale input data')
    parser.add_argument('--stop_loss', type=int, default=0, help='stop loss parameter')
    
    parser.add_argument('--initial_temp', type=float, default=1.0)
    parser.add_argument('--final_temp', type=float, default=0.1)
    parser.add_argument('--anneal_epochs', type=int, default=5)
    parser.add_argument('--schedule_type', type=str, default="linear")
    parser.add_argument('--lambda_KL', type=float, default=0.01)
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--max_grad_norm', type=float, default=5.0)
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = f"./xai_results_{args.model_id}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    args_dict = vars(args)
    model_args = dotdict(args_dict)
    model_args.device = device
    
    model = load_model_from_checkpoint(args.checkpoint, model_args, device)
    
    if model is None:
        print("Failed to load model. Exiting.")
        sys.exit(1)
    
    _, data_loader = data_provider(model_args, flag='test')    
    try:
        data_set, _ = data_provider(model_args, flag='test')
        feature_names = extract_feature_names(data_set, args.dim_input)
    except Exception as e:
        print(f"Error extracting feature names: {e}")
        feature_names = [f"Feature_{i}" for i in range(args.dim_input)]
        
    class_names = ['SELL', 'BUY'] if args.num_classes == 2 else [f"Class_{i}" for i in range(args.num_classes)]
    xai = TreeXAI(model, device, feature_names, class_names)
    
    try:
        results = xai.analyze(
            data_loader,
            output_dir=args.output_dir,
            threshold_fs=args.threshold_fs,
            top_k=args.top_k
        )
        
        print(f"XAI analysis complete. Results saved to {args.output_dir}")
        print(f"Overall metrics: Accuracy={results['metrics']['accuracy']:.4f}, "
              f"MCC={results['metrics']['mcc']:.4f}, F1={results['metrics']['f1_score']:.4f}")
        
        global_fi = results['global_feature_importance']
        top_indices = np.argsort(global_fi)[-10:]  # Top 10 features
        
        print("\nTop 10 Global Features:")
        for idx in reversed(top_indices):
            print(f"  {feature_names[idx]}: {global_fi[idx]:.4f}")
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()