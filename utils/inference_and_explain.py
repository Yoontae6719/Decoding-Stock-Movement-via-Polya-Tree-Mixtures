import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef


class Dataset_SNP_XAI(Dataset):

    def __init__(
        self,
        args,
        root_path,
        data_path='SNP.csv',
        flag='test',
        scale=True,
        stop_loss=0,
        stock_name='AAPL'
    ):
        super().__init__()
        assert flag in ['train','val','test']
        self.args = args
        self.root_path = root_path
        self.data_path = data_path
        self.scale = scale
        self.stop_loss = stop_loss
        self.stock_name = stock_name

        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw['Date'] = pd.to_datetime(df_raw['Date'])
        df_raw = df_raw.dropna()

        df_raw[['Y','Y_2','Y_3','Y_4','Y_5']] = df_raw[['Y','Y_2','Y_3','Y_4','Y_5']].apply(
            lambda col: col.map({'SELL':0, 'BUY':1})
        )

        col_drop = ['Stock','Date','Y','Y_2','Y_3','Y_4','Y_5']
        df_x_all = df_raw.drop(columns=col_drop)
        self.feature_names = df_x_all.columns.tolist()

        mask_train = (df_raw['Date']>='2020-01-01') & (df_raw['Date']<='2022-12-31')
        train_x = df_x_all[mask_train].values

        if self.scale:
            quantile_train = train_x.astype(np.float64)
            stds = np.std(quantile_train, axis=0, keepdims=True)
            noise_std = 1e-3 / np.maximum(stds, 1e-3)
            quantile_train += noise_std * np.random.randn(*quantile_train.shape)

            self.scaler = QuantileTransformer(output_distribution='normal', random_state=42)
            self.scaler.fit(quantile_train)
            data_x_all = self.scaler.transform(df_x_all.values)
        else:
            self.scaler = None
            data_x_all = df_x_all.values

        df_sub = df_raw[df_raw['Stock']==self.stock_name].copy()
        sub_idx = df_sub.index
        data_x_sub = data_x_all[sub_idx]

        mask_train_sub = (df_sub['Date']>='2020-01-01') & (df_sub['Date']<='2022-12-31')
        mask_val_sub   = (df_sub['Date']>='2023-01-01') & (df_sub['Date']<='2023-12-31')
        mask_test_sub  = (df_sub['Date']>='2024-01-01')

        num_train = mask_train_sub.sum()
        num_val   = mask_val_sub.sum()
        num_test  = mask_test_sub.sum()

        border1s = [0, num_train, num_train+num_val]
        border2s = [num_train, num_train+num_val, num_train+num_val+num_test]

        if self.stop_loss==0:
            df_sub_y = df_sub['Y'].values
        elif self.stop_loss==2:
            df_sub_y = df_sub['Y_2'].values
        elif self.stop_loss==3:
            df_sub_y = df_sub['Y_3'].values
        elif self.stop_loss==4:
            df_sub_y = df_sub['Y_4'].values
        elif self.stop_loss==5:
            df_sub_y = df_sub['Y_5'].values
        else:
            raise ValueError("stop_loss must be in [0,2,3,4,5].")

        set_type = self.set_type
        border1, border2 = border1s[set_type], border2s[set_type]

        self.data_x = data_x_sub[border1:border2]
        self.data_y = df_sub_y[border1:border2]

    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = self.data_y[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.data_x)



def print_confusion_matrix(labels, preds, title="Confusion Matrix"):
    cm = confusion_matrix(labels, preds)
    print(f"\n[{title}]")
    print(cm)
    print("\n[Classification Report]")
    print(classification_report(labels, preds, digits=4))
    mcc = matthews_corrcoef(labels, preds)
    print(f"MCC: {mcc:.4f}")



def gather_routing_statistics_hard(node, x, y, device, path="root"):
    n_samples = x.size(0)
    if n_samples==0:
        return [{
            'node_path': path,
            'n_samples': 0,
            'n_label0': 0,
            'n_label1': 0,
            'acc_leaf': None,
            'is_leaf': node.is_leaf
        }]

    n_label0 = (y==0).sum().item()
    n_label1 = (y==1).sum().item()

    if node.is_leaf:
        with torch.no_grad():
            gamma, _ = node.feature_selector(temperature=1.0, training=False)
            logits = node.local_expert(x, gamma)
            preds = torch.argmax(logits, dim=1)
        correct = (preds==y).sum().item()
        acc_leaf = correct/n_samples
        return [{
            'node_path': path,
            'n_samples': n_samples,
            'n_label0': n_label0,
            'n_label1': n_label1,
            'acc_leaf': acc_leaf,
            'is_leaf': True,
        }]
    else:
        if node.gating is not None:
            with torch.no_grad():
                gate_logit = node.gating(x).squeeze(-1)
            p_left = torch.sigmoid(gate_logit)
        else:
            with torch.no_grad():
                gate_logit = x @ node.w + node.b
            p_left = torch.sigmoid(gate_logit)
        mask_left = (p_left>0.5)
        mask_right= ~mask_left

        x_left, y_left = x[mask_left], y[mask_left]
        x_right,y_right= x[mask_right],y[mask_right]

        cur_info = {
            'node_path': path,
            'n_samples': n_samples,
            'n_label0': n_label0,
            'n_label1': n_label1,
            'acc_leaf': None,
            'is_leaf': False,
        }

        L_list = gather_routing_statistics_hard(node.left_child, x_left, y_left, device, path+"_L")
        R_list = gather_routing_statistics_hard(node.right_child, x_right, y_right, device, path+"_R")
        return [cur_info] + L_list + R_list


def get_all_leaves_in_order(node):
    if node.is_leaf:
        return [node]
    else:
        return get_all_leaves_in_order(node.left_child)+get_all_leaves_in_order(node.right_child)

def compute_leaf_distribution_soft(node, x, temperature=0.1):
    if node.is_leaf:
        return torch.ones(x.size(0), 1, device=x.device)
    else:
        if node.gating is not None:
            gate_logit = node.gating(x).squeeze(-1)
        else:
            gate_logit = x @ node.w + node.b

        p_left = torch.sigmoid(gate_logit)
        dist_left = compute_leaf_distribution_soft(node.left_child, x, temperature)
        dist_right= compute_leaf_distribution_soft(node.right_child, x, temperature)

        w_left = dist_left * p_left.unsqueeze(-1)
        w_right= dist_right* (1-p_left).unsqueeze(-1)
        return torch.cat([w_left, w_right], dim=1)


def gather_leaf_feature_phi(node, path="root"):
    if node.is_leaf:
        with torch.no_grad():
            phi = torch.sigmoid(node.feature_selector.logit_phi).cpu().numpy()
        return [(path, phi)]
    else:
        L = gather_leaf_feature_phi(node.left_child, path+"_L")
        R = gather_leaf_feature_phi(node.right_child, path+"_R")
        return L + R

def gather_leaf_feature_scores(node, path="root"):
    if node.is_leaf:
        with torch.no_grad():
            phi = torch.sigmoid(node.feature_selector.logit_phi)
            score = torch.log(phi+1e-9)-torch.log(1-phi+1e-9)
            score_np = score.cpu().numpy()
        return [(path, score_np)]
    else:
        L = gather_leaf_feature_scores(node.left_child, path+"_L")
        R = gather_leaf_feature_scores(node.right_child, path+"_R")
        return L + R


def compute_global_importance_hard(model, x, device):
    node_id2path = {}
    def build_map(node, path="root"):
        node_id2path[id(node)] = path
        if not node.is_leaf:
            build_map(node.left_child, path+"_L")
            build_map(node.right_child, path+"_R")
    build_map(model.root, "root")

    N = x.size(0)
    idx_all = torch.arange(N, device=device)

    def route_samples(node, x_cur, idx_cur):
        if x_cur.size(0)==0:
            return {}
        if node.is_leaf:
            return {id(node): idx_cur}
        else:
            if node.gating is not None:
                gate_logit = node.gating(x_cur).squeeze(-1)
            else:
                gate_logit = x_cur @ node.w + node.b
            p_left = torch.sigmoid(gate_logit)
            mask_left = (p_left>0.5)
            mask_right= ~mask_left
            x_left, i_left = x_cur[mask_left], idx_cur[mask_left]
            x_right,i_right= x_cur[mask_right], idx_cur[mask_right]
            dL= route_samples(node.left_child, x_left, i_left)
            dR= route_samples(node.right_child,x_right,i_right)
            merged={}
            for k,v in dL.items():
                merged[k]=v
            for k,v in dR.items():
                if k in merged:
                    merged[k]=torch.cat([merged[k],v],dim=0)
                else:
                    merged[k]=v
            return merged

    leaf_map = route_samples(model.root, x, idx_all)

    leaf_phi_list = gather_leaf_feature_phi(model.root,"root")
    path2phi = {lp[0]: lp[1] for lp in leaf_phi_list}

    dim_input = model.dim_input
    global_imp = np.zeros(dim_input, dtype=np.float32)
    coverage_dict = {}

    for node_id, idxes in leaf_map.items():
        path = node_id2path[node_id]
        coverage = float(idxes.size(0))/float(N)
        coverage_dict[path] = coverage
        phi_leaf = path2phi[path]
        global_imp += coverage*phi_leaf

    return global_imp, coverage_dict

def compute_global_importance_soft(model, x, device, temperature=0.1):
    leaves = get_all_leaves_in_order(model.root)
    with torch.no_grad():
        dist = compute_leaf_distribution_soft(model.root, x, temperature) # (N,L)
    coverage_vec = dist.sum(dim=0)/ x.size(0) # (L,)

    node_id2path = {}
    def build_map(node, path="root"):
        node_id2path[id(node)] = path
        if not node.is_leaf:
            build_map(node.left_child, path+"_L")
            build_map(node.right_child, path+"_R")
    build_map(model.root,"root")

    global_imp = np.zeros(model.dim_input, dtype=np.float32)
    coverage_dict = {}

    for i, leaf_node in enumerate(leaves):
        path = node_id2path[id(leaf_node)]
        cov_i = coverage_vec[i].item()
        coverage_dict[path] = cov_i
        with torch.no_grad():
            phi_leaf = torch.sigmoid(leaf_node.feature_selector.logit_phi).cpu().numpy()
        global_imp += cov_i*phi_leaf

    return global_imp, coverage_dict


def print_leaf_top_bottom_features(node, feature_names, top_k=5, bottom_k=5):
    leaf_scores_list = gather_leaf_feature_scores(node, "root")
    print("\n=== Leaf Feature Selection (Top/Bottom K) ===")
    for (path, score_arr) in leaf_scores_list:
        idx_sorted = np.argsort(score_arr)
        idx_bottom = idx_sorted[:bottom_k]
        idx_top    = idx_sorted[-top_k:]

        print(f"\n[Leaf = {path}]")
        print(" - Top Features:")
        for i in reversed(idx_top):
            print(f"   {feature_names[i]} => score={score_arr[i]:.3f}")
        print(" - Bottom Features:")
        for i in idx_bottom:
            print(f"   {feature_names[i]} => score={score_arr[i]:.3f}")

def print_global_top_bottom_features(global_imp, feature_names, top_k=5, bottom_k=5):
    idx_sorted = np.argsort(global_imp)
    idx_bottom = idx_sorted[:bottom_k]
    idx_top    = idx_sorted[-top_k:]

    print(f"\n=== Global Feature Importance (Top {top_k}, Bottom {bottom_k}) ===")
    print("[Top Features]")
    for i in reversed(idx_top):
        print(f"   {feature_names[i]} => {global_imp[i]:.4f}")
    print("[Bottom Features]")
    for i in idx_bottom:
        print(f"   {feature_names[i]} => {global_imp[i]:.4f}")


def inference_and_explain(
    args,
    model,
    test_loader,
    gating_mode='hard',     # 'hard' | 'soft'
    temperature=0.1,       
    top_k_leaf=5,
    bottom_k_leaf=5,
    do_global_importance=True,
    top_k_global=5,
    bottom_k_global=5
):
    device = getattr(args,'device','cpu')
    model.eval()

    preds_list, labels_list = [], []
    all_x, all_y = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).squeeze(-1)

            if gating_mode=='hard':
                probs = model.forward_hard(x_batch)
            else:
                probs = model(x_batch, temperature=temperature)

            preds = torch.argmax(probs, dim=1)
            preds_list.append(preds.cpu().numpy())
            labels_list.append(y_batch.cpu().numpy())

            all_x.append(x_batch)
            all_y.append(y_batch)

    preds_array = np.concatenate(preds_list, axis=0)
    labels_array= np.concatenate(labels_list, axis=0)
    acc = (preds_array==labels_array).mean()

    print(f"\n===== Test Accuracy ({gating_mode} gating) = {acc:.4f} =====")
    print_confusion_matrix(labels_array, preds_array, title=f"Confusion Matrix ({gating_mode} gating)")

    X_cat = torch.cat(all_x, dim=0)
    Y_cat = torch.cat(all_y, dim=0)
    if gating_mode=='hard':
        routing_info = gather_routing_statistics_hard(model.root, X_cat, Y_cat, device, "root")
        print("\n=== Node-level Routing Statistics (Hard Gating) ===")
        for ri in routing_info:
            if ri['is_leaf']:
                acc_str = f"{ri['acc_leaf']:.4f}" if ri['acc_leaf'] else "NA"
                print(f"[Leaf] {ri['node_path']} | #samples={ri['n_samples']} "
                      f"| label0={ri['n_label0']}, label1={ri['n_label1']}, leaf_acc={acc_str}")
            else:
                print(f"[Node] {ri['node_path']} | #samples={ri['n_samples']} "
                      f"| label0={ri['n_label0']}, label1={ri['n_label1']}")

    # Leaf Top/Bottom
    if hasattr(test_loader.dataset, 'feature_names'):
        feature_names = test_loader.dataset.feature_names
    else:
        feature_names = [f"feat_{i}" for i in range(model.dim_input)]
    print_leaf_top_bottom_features(model.root, feature_names, top_k=top_k_leaf, bottom_k=bottom_k_leaf)

    # Global Feature Importance
    if do_global_importance:
        if gating_mode=='hard':
            global_imp, coverage_dict = compute_global_importance_hard(model, X_cat, device)
        else:
            global_imp, coverage_dict = compute_global_importance_soft(model, X_cat, device, temperature=temperature)

        print("\n=== Leaf Coverage ===")
        for path, cov in coverage_dict.items():
            print(f"   {path} => coverage={cov:.4f}")

        print_global_top_bottom_features(global_imp, feature_names, top_k=top_k_global, bottom_k=bottom_k_global)

    print(f"\n>>> Inference & Explain Done. (mode={gating_mode})")


