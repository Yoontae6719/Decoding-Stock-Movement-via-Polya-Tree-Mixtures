import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef
import torch.nn.functional as F


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

        # 특정 종목만
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


###############################################################################
# AAA Node/leaf routing stat (Hard/Soft)
###############################################################################

def gather_routing_statistics_hard(node, x, y, device, path="root"):
    n_samples = x.size(0)
    if n_samples == 0:
        return [{
            'node_path': path,
            'n_samples': 0,
            'n_label0': 0,
            'n_label1': 0,
            'acc_leaf': None,
            'mcc_leaf': None,
            'is_leaf': node.is_leaf
        }]

    n_label0 = (y == 0).sum().item()
    n_label1 = (y == 1).sum().item()

    if node.is_leaf:
        with torch.no_grad():
            gamma, _ = node.feature_selector(temperature=1.0, training=False)
            logits = node.local_expert(x, gamma)
            preds = torch.argmax(logits, dim=1)

        correct = (preds == y).sum().item()
        acc_leaf = correct / n_samples if n_samples>0 else None

        if n_samples > 1:
            try:
                mcc_leaf = matthews_corrcoef(y.cpu().numpy(), preds.cpu().numpy())
            except:
                mcc_leaf = 0.0
        else:
            mcc_leaf = 0.0

        return [{
            'node_path': path,
            'n_samples': n_samples,
            'n_label0': n_label0,
            'n_label1': n_label1,
            'acc_leaf': acc_leaf,
            'mcc_leaf': mcc_leaf,
            'is_leaf': True,
        }]
    else:
        if node.gating is not None:
            with torch.no_grad():
                gate_logit = node.gating(x).squeeze(-1)
        else:
            with torch.no_grad():
                gate_logit = x @ node.w + node.b

        p_left = torch.sigmoid(gate_logit)
        mask_left = (p_left > 0.5)
        mask_right = ~mask_left

        x_left, y_left = x[mask_left], y[mask_left]
        x_right, y_right = x[mask_right], y[mask_right]

        cur_info = {
            'node_path': path,
            'n_samples': n_samples,
            'n_label0': n_label0,
            'n_label1': n_label1,
            'acc_leaf': None,
            'mcc_leaf': None,
            'is_leaf': False,
        }

        L_list = gather_routing_statistics_hard(node.left_child, x_left, y_left, device, path+"_L")
        R_list = gather_routing_statistics_hard(node.right_child, x_right, y_right, device, path+"_R")

        return [cur_info] + L_list + R_list


def gather_routing_statistics_soft(node, x, y, coverage, device, path="root"):
    n_coverage = coverage.sum().item()
    if n_coverage < 1e-9:
        return [{
            'node_path': path,
            'coverage': 0.0,
            'coverage_label0': 0.0,
            'coverage_label1': 0.0,
            'acc_leaf': None,
            'mcc_leaf': None,
            'is_leaf': node.is_leaf,
        }]

    coverage_label0 = coverage[y == 0].sum().item()
    coverage_label1 = coverage[y == 1].sum().item()

    if node.is_leaf:
        with torch.no_grad():
            gamma, _ = node.feature_selector(temperature=1.0, training=False)
            logits = node.local_expert(x, gamma)
            preds = torch.argmax(logits, dim=1)

        # coverage-weighted accuracy
        correct_mask = (preds == y).float()
        coverage_correct = (coverage * correct_mask).sum().item()
        acc_leaf = coverage_correct / (n_coverage + 1e-9)

        # coverage-weighted MCC (weighted confusion matrix)
        cm = np.zeros((2, 2), dtype=np.float64)
        preds_np = preds.cpu().numpy()
        y_np = y.cpu().numpy()
        coverage_np = coverage.cpu().numpy()

        for i in range(x.size(0)):
            w = coverage_np[i]
            if w < 1e-12:
                continue
            pi = preds_np[i]
            yi = y_np[i]
            if pi == 0 and yi == 0:
                cm[0, 0] += w
            elif pi == 1 and yi == 1:
                cm[1, 1] += w
            elif pi == 1 and yi == 0:
                cm[0, 1] += w
            elif pi == 0 and yi == 1:
                cm[1, 0] += w

        TN, FP = cm[0, 0], cm[0, 1]
        FN, TP = cm[1, 0], cm[1, 1]
        numerator = (TP * TN - FP * FN)
        denominator = np.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN)) + 1e-15
        mcc_leaf = numerator / denominator

        return [{
            'node_path': path,
            'coverage': n_coverage,
            'coverage_label0': coverage_label0,
            'coverage_label1': coverage_label1,
            'acc_leaf': acc_leaf,
            'mcc_leaf': mcc_leaf,
            'is_leaf': True,
        }]
    else:
        if node.gating is not None:
            with torch.no_grad():
                gate_logit = node.gating(x).squeeze(-1)
        else:
            with torch.no_grad():
                gate_logit = x @ node.w + node.b
        p_left = torch.sigmoid(gate_logit)
        p_right= 1 - p_left

        coverage_left  = coverage * p_left
        coverage_right = coverage * p_right

        cur_info = {
            'node_path': path,
            'coverage': n_coverage,
            'coverage_label0': coverage_label0,
            'coverage_label1': coverage_label1,
            'acc_leaf': None,
            'mcc_leaf': None,
            'is_leaf': False,
        }

        L_list = gather_routing_statistics_soft(node.left_child, x, y, coverage_left, device, path+"_L")
        R_list = gather_routing_statistics_soft(node.right_child, x, y, coverage_right, device, path+"_R")

        return [cur_info] + L_list + R_list


###############################################################################
# BBB Leaf phi log-odds
###############################################################################

def gather_leaf_feature_scores(node, path="root"):
    """
    leaf별로 phi의 log-odds(= log(phi) - log(1-phi))를 반환
    """
    if node.is_leaf:
        with torch.no_grad():
            phi = torch.sigmoid(node.feature_selector.logit_phi)
            score = torch.log(phi+1e-9) - torch.log(1 - phi + 1e-9)
            score_np = score.cpu().numpy()
        return [(path, score_np)]
    else:
        L = gather_leaf_feature_scores(node.left_child, path+"_L")
        R = gather_leaf_feature_scores(node.right_child, path+"_R")
        return L + R

def print_leaf_top_bottom_features(node, feature_names, top_k=5, bottom_k=5):
    leaf_scores_list = gather_leaf_feature_scores(node, "root")
    print("\n=== Leaf Feature Selection (Top/Bottom K by phi log-odds) ===")
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


def gather_leaf_feature_selection_data(node, feature_names, top_k=5, bottom_k=5):
    """
    Leaf phi log-odds Top/Bottom K information
    """
    leaf_scores_list = gather_leaf_feature_scores(node, "root")
    leaf_features_data = []
    for (path, score_arr) in leaf_scores_list:
        idx_sorted = np.argsort(score_arr)
        idx_bottom = idx_sorted[:bottom_k]
        idx_top    = idx_sorted[-top_k:]
        
        top_feats = []
        bottom_feats = []
        for i in reversed(idx_top):
            top_feats.append((feature_names[i], float(score_arr[i])))
        for i in idx_bottom:
            bottom_feats.append((feature_names[i], float(score_arr[i])))

        leaf_features_data.append({
            'leaf_path': path,
            'top_features': top_feats,
            'bottom_features': bottom_feats,
        })
    return leaf_features_data


###############################################################################
# BBB2 Global Importance (기존: coverage * phi)
###############################################################################

def compute_global_importance_hard(model, x, device):
    # Hard gating: coverage * phi
    N = x.size(0)
    idx_all = torch.arange(N, device=device)

    node_id2path = {}
    def build_map(n, path="root"):
        node_id2path[id(n)] = path
        if not n.is_leaf:
            build_map(n.left_child, path+"_L")
            build_map(n.right_child, path+"_R")
    build_map(model.root, "root")

    leaf_map = {}
    def route_samples(node, x_cur, idx_cur):
        if x_cur.size(0)==0:
            return
        if node.is_leaf:
            leaf_map[id(node)] = leaf_map.get(id(node), []) + [idx_cur]
        else:
            if node.gating is not None:
                gate_logit = node.gating(x_cur).squeeze(-1)
            else:
                gate_logit = x_cur @ node.w + node.b
            p_left = torch.sigmoid(gate_logit)
            mask_left = (p_left>0.5)
            mask_right= ~mask_left
            route_samples(node.left_child, x_cur[mask_left], idx_cur[mask_left])
            route_samples(node.right_child, x_cur[mask_right], idx_cur[mask_right])

    route_samples(model.root, x, idx_all)

    dim_input = model.dim_input
    global_imp = np.zeros(dim_input, dtype=np.float32)
    coverage_dict = {}

    def find_node_by_id(nid, root):
        if id(root) == nid:
            return root
        if not root.is_leaf:
            lf = find_node_by_id(nid, root.left_child)
            if lf is not None: return lf
            rf = find_node_by_id(nid, root.right_child)
            if rf is not None: return rf
        return None

    for nid, idxes_list in leaf_map.items():
        idx_cat = torch.cat(idxes_list, dim=0)
        coverage = float(idx_cat.size(0))/float(N)
        path = node_id2path[nid]
        coverage_dict[path] = coverage

        leaf_node = find_node_by_id(nid, model.root)
        with torch.no_grad():
            phi_leaf = torch.sigmoid(leaf_node.feature_selector.logit_phi).cpu().numpy()

        global_imp += coverage * phi_leaf

    return global_imp, coverage_dict


def compute_global_importance_soft(model, x, device, temperature=0.1):
    # Soft gating: coverage * phi
    def get_leaves_in_order(n):
        if n.is_leaf:
            return [n]
        else:
            return get_leaves_in_order(n.left_child)+get_leaves_in_order(n.right_child)

    leaves = get_leaves_in_order(model.root)
    node_id2path = {}
    def build_map(n, path="root"):
        node_id2path[id(n)] = path
        if not n.is_leaf:
            build_map(n.left_child, path+"_L")
            build_map(n.right_child, path+"_R")
    build_map(model.root,"root")

    def compute_leaf_dist(n, x_):
        if n.is_leaf:
            return torch.ones(x_.size(0),1, device=device)
        else:
            if n.gating is not None:
                gate_logit = n.gating(x_).squeeze(-1)
            else:
                gate_logit = x_ @ n.w + n.b
            p_left = torch.sigmoid(gate_logit)
            left_ = compute_leaf_dist(n.left_child, x_)
            right_= compute_leaf_dist(n.right_child, x_)
            return torch.cat([left_*p_left.unsqueeze(-1), right_*(1-p_left).unsqueeze(-1)], dim=1)

    with torch.no_grad():
        dist = compute_leaf_dist(model.root, x)  # (N, #leaves)
    coverage_vec = dist.sum(dim=0)/ x.size(0)

    dim_input = model.dim_input
    global_imp = np.zeros(dim_input, dtype=np.float32)
    coverage_dict = {}

    for i, lf in enumerate(leaves):
        path = node_id2path[id(lf)]
        cov_i = coverage_vec[i].item()
        coverage_dict[path] = cov_i
        with torch.no_grad():
            phi = torch.sigmoid(lf.feature_selector.logit_phi).cpu().numpy()
        global_imp += cov_i * phi

    return global_imp, coverage_dict


###############################################################################
# (BBB3) Global Influence (새 방식): coverage * (avg_gamma * MLP_weight)
###############################################################################

def compute_local_expert_weight_influence(local_expert):
    """
      w_infl[i] = sum_{j=1..hidden_dim} sum_{c=1..num_classes}
                  (|W1[j,i]| * |W2[c,j]|)
    """
    fc1_w = local_expert.fc1.weight.detach()  # shape (hidden_dim, D)
    fc2_w = local_expert.fc2.weight.detach()  # shape (num_classes, hidden_dim)

    hidden_dim, D_ = fc1_w.shape
    num_classes = fc2_w.shape[0]

    infl = torch.zeros(D_, dtype=torch.float32, device=fc1_w.device)
    abs_fc1 = fc1_w.abs()
    abs_fc2 = fc2_w.abs()

    for i in range(D_):
        val_i = 0.0
        for j in range(hidden_dim):
            for c in range(num_classes):
                val_i += abs_fc1[j,i].item() * abs_fc2[c,j].item()
        infl[i] = val_i
    return infl


def compute_global_influence_hard(model, x, device):
    """
    Hard gating: coverage * (avg_gamma * MLP_weight_influence)
    """
    N = x.size(0)
    idx_all = torch.arange(N, device=device)

    node_id2path = {}
    def build_map(n, path="root"):
        node_id2path[id(n)] = path
        if not n.is_leaf:
            build_map(n.left_child, path+"_L")
            build_map(n.right_child, path+"_R")
    build_map(model.root, "root")

    leaf_map = {}
    def route(node, x_cur, idx_cur):
        if x_cur.size(0)==0:
            return
        if node.is_leaf:
            leaf_map[id(node)] = leaf_map.get(id(node), []) + [idx_cur]
        else:
            if node.gating is not None:
                gate_logit = node.gating(x_cur).squeeze(-1)
            else:
                gate_logit = x_cur @ node.w + node.b
            p_left = torch.sigmoid(gate_logit)
            mask_left = (p_left>0.5)
            route(node.left_child, x_cur[mask_left], idx_cur[mask_left])
            route(node.right_child,x_cur[~mask_left], idx_cur[~mask_left])

    route(model.root, x, idx_all)

    coverage_dict = {}
    for nid, idxes in leaf_map.items():
        path = node_id2path[nid]
        idx_cat = torch.cat(idxes, dim=0)
        coverage_dict[path] = float(idx_cat.size(0))/ float(N)

    node_id2gamma_list = {}
    def init_dict(n):
        node_id2gamma_list[id(n)] = []
        if not n.is_leaf:
            init_dict(n.left_child)
            init_dict(n.right_child)
    init_dict(model.root)

    def collect_gamma(node, x_cur):
        if node.is_leaf:
            with torch.no_grad():
                gamma, _ = node.feature_selector(temperature=1.0, training=False)
            node_id2gamma_list[id(node)].append(gamma.cpu().numpy())
        else:
            if node.gating is not None:
                gate_logit = node.gating(x_cur).squeeze(-1)
            else:
                gate_logit = x_cur @ node.w + node.b
            p_left = torch.sigmoid(gate_logit)
            mask_left = (p_left>0.5)
            collect_gamma(node.left_child, x_cur[mask_left])
            collect_gamma(node.right_child,x_cur[~mask_left])

    collect_gamma(model.root, x)

    node_id2weight = {}
    def collect_weight(n):
        if n.is_leaf:
            w_infl = compute_local_expert_weight_influence(n.local_expert)
            node_id2weight[id(n)] = w_infl.cpu().numpy()
        else:
            collect_weight(n.left_child)
            collect_weight(n.right_child)
    collect_weight(model.root)

    D = model.dim_input
    global_infl = np.zeros(D, dtype=np.float32)

    for nid, gammas in node_id2gamma_list.items():
        if len(gammas)==0:
            continue
        gamma_array = np.stack(gammas, axis=0)
        avg_gamma = gamma_array.mean(axis=0)
        w_infl = node_id2weight.get(nid,None)
        if w_infl is None:
            continue
        path = node_id2path[nid]
        cov = coverage_dict.get(path, 0.0)
        leaf_infl = avg_gamma * w_infl
        global_infl += cov * leaf_infl

    return global_infl, coverage_dict


def compute_global_influence_soft(model, x, device, temperature=0.1):
    """
    Soft gating: coverage * (avg_gamma * MLP_weight_influence)
    """
    def get_leaves_in_order(n):
        if n.is_leaf:
            return [n]
        else:
            return get_leaves_in_order(n.left_child) + get_leaves_in_order(n.right_child)
    leaves = get_leaves_in_order(model.root)

    node_id2path = {}
    def build_map(n, path="root"):
        node_id2path[id(n)] = path
        if not n.is_leaf:
            build_map(n.left_child, path+"_L")
            build_map(n.right_child, path+"_R")
    build_map(model.root, "root")

    def compute_leaf_dist(node, x_):
        if node.is_leaf:
            return torch.ones(x_.size(0),1, device=device)
        else:
            if node.gating is not None:
                gate_logit = node.gating(x_).squeeze(-1)
            else:
                gate_logit = x_ @ node.w + node.b
            p_left = torch.sigmoid(gate_logit)
            left_ = compute_leaf_dist(node.left_child, x_)
            right_= compute_leaf_dist(node.right_child, x_)
            return torch.cat([left_*p_left.unsqueeze(-1), right_*(1-p_left).unsqueeze(-1)], dim=1)

    with torch.no_grad():
        dist = compute_leaf_dist(model.root, x)  # (N, #leaves)
    coverage_vec = dist.sum(dim=0)/ x.size(0)

    # leaf별 평균 gamma
    node_id2gammas = {}
    def init_dict(n):
        node_id2gammas[id(n)] = []
        if not n.is_leaf:
            init_dict(n.left_child)
            init_dict(n.right_child)
    init_dict(model.root)

    def collect_leaf_gamma_soft(node, x_, coverage):
        if node.is_leaf:
            with torch.no_grad():
                phi_vec = torch.sigmoid(node.feature_selector.logit_phi).cpu().numpy()
            cov_sum = coverage.sum().item()
            if cov_sum>1e-12:
                node_id2gammas[id(node)].append((phi_vec, cov_sum))
            return
        else:
            if node.gating is not None:
                gate_logit = node.gating(x_).squeeze(-1)
            else:
                gate_logit = x_ @ node.w + node.b
            p_left = torch.sigmoid(gate_logit)
            coverage_left = coverage * p_left
            coverage_right= coverage * (1-p_left)
            collect_leaf_gamma_soft(node.left_child, x_, coverage_left)
            collect_leaf_gamma_soft(node.right_child, x_, coverage_right)

    coverage_init = torch.ones(x.size(0), device=device)
    collect_leaf_gamma_soft(model.root, x, coverage_init)

    node_id2weight = {}
    def collect_weight(n):
        if n.is_leaf:
            w_infl = compute_local_expert_weight_influence(n.local_expert)
            node_id2weight[id(n)] = w_infl.cpu().numpy()
        else:
            collect_weight(n.left_child)
            collect_weight(n.right_child)
    collect_weight(model.root)

    D = model.dim_input
    global_infl = np.zeros(D, dtype=np.float32)
    coverage_dict = {}

    for i, lf in enumerate(leaves):
        cov_i = coverage_vec[i].item()
        path = node_id2path[id(lf)]
        coverage_dict[path] = cov_i

        items = node_id2gammas[id(lf)]
        if len(items)==0:
            continue
        phi_sum = np.zeros(D, dtype=np.float32)
        w_sum = 0.0
        for (ph, w_) in items:
            phi_sum += ph*w_
            w_sum += w_
        avg_gamma = phi_sum/(w_sum+1e-9)

        w_infl = node_id2weight[id(lf)]
        leaf_infl = avg_gamma * w_infl
        global_infl += cov_i * leaf_infl

    return global_infl, coverage_dict


###############################################################################
# (CCC) Leaf-level (avg_gamma x MLP weight) (데이터만 수집)
###############################################################################

def collect_leaf_influence_gamma_weight(
    model,
    x,            # (N, D)
    feature_names,
    gating_mode='hard',
    temperature=0.1,
    top_k=5,
    bottom_k=5,
    device='cpu'
):

    model.eval()
    x = x.to(device)
    N, D = x.shape
    if len(feature_names) != D:
        feature_names = [f"feat_{i}" for i in range(D)]

    def get_all_leaves_in_order(n, path="root"):
        leaves = []
        def rec(node_, p_):
            if node_.is_leaf:
                leaves.append((node_, p_))
            else:
                rec(node_.left_child, p_+"_L")
                rec(node_.right_child, p_+"_R")
        rec(n, path)
        return leaves
    leaves_in_order = get_all_leaves_in_order(model.root, "root")

    # MLP weight 영향도
    def compute_local_expert_weight_influence(local_expert):
        fc1_w = local_expert.fc1.weight.detach()
        fc2_w = local_expert.fc2.weight.detach()
        hdim, D_ = fc1_w.shape
        cdim = fc2_w.shape[0]
        abs_fc1 = fc1_w.abs()
        abs_fc2 = fc2_w.abs()
        out = torch.zeros(D_, dtype=torch.float32, device=fc1_w.device)
        for i in range(D_):
            val_i = 0.0
            for j in range(hdim):
                for c in range(cdim):
                    val_i += abs_fc1[j,i].item()*abs_fc2[c,j].item()
            out[i] = val_i
        return out

    leaf_influence_data = []


    if gating_mode=='hard':
        node_id2idxes = {}
        def init_dict(n):
            node_id2idxes[id(n)] = []
            if not n.is_leaf:
                init_dict(n.left_child)
                init_dict(n.right_child)
        init_dict(model.root)

        def route_samples(node, x_cur, idx_cur):
            if x_cur.size(0)==0:
                return
            if node.is_leaf:
                node_id2idxes[id(node)].append(idx_cur)
            else:
                if node.gating is not None:
                    gate_logit = node.gating(x_cur).squeeze(-1)
                else:
                    gate_logit = x_cur @ node.w + node.b
                p_left = torch.sigmoid(gate_logit)
                mask_left = (p_left>0.5)
                route_samples(node.left_child, x_cur[mask_left], idx_cur[mask_left])
                route_samples(node.right_child, x_cur[~mask_left], idx_cur[~mask_left])

        idx_all = torch.arange(N, device=device)
        route_samples(model.root, x, idx_all)

        node_id2avg_gamma = {}
        for (leaf, path) in leaves_in_order:
            idx_list = node_id2idxes[id(leaf)]
            if len(idx_list)==0:
                node_id2avg_gamma[id(leaf)] = np.zeros(D, dtype=np.float32)
                continue

            idx_cat = torch.cat(idx_list, dim=0)
            n_leaf = idx_cat.size(0)
            if n_leaf>0:
                with torch.no_grad():
                    gamma_leaf, _ = leaf.feature_selector(temperature=1.0, training=False)
                avg_gamma = gamma_leaf.cpu().numpy()  # (D,)
            else:
                avg_gamma = np.zeros(D, dtype=np.float32)

            node_id2avg_gamma[id(leaf)] = avg_gamma

        for (leaf, path) in leaves_in_order:
            avg_gamma_leaf = node_id2avg_gamma[id(leaf)]
            w_infl = compute_local_expert_weight_influence(leaf.local_expert).cpu().numpy()
            leaf_infl = avg_gamma_leaf * w_infl

            idxes_list = node_id2idxes[id(leaf)]
            n_leaf_smp = sum([it_.size(0) for it_ in idxes_list])
            coverage_leaf = float(n_leaf_smp)/float(N)

            # 정렬
            idx_sorted = np.argsort(leaf_infl)
            idx_bottom = idx_sorted[:bottom_k]
            idx_top    = idx_sorted[-top_k:]

            top_features_info = []
            bottom_features_info = []
            for i in reversed(idx_top):
                top_features_info.append((
                    feature_names[i],
                    float(leaf_infl[i]),
                    float(avg_gamma_leaf[i]),
                    float(w_infl[i])
                ))
            for i in idx_bottom:
                bottom_features_info.append((
                    feature_names[i],
                    float(leaf_infl[i]),
                    float(avg_gamma_leaf[i]),
                    float(w_infl[i])
                ))

            leaf_influence_data.append({
                "leaf_path": path,
                "coverage": coverage_leaf,
                "top_features": top_features_info,
                "bottom_features": bottom_features_info,
            })

    else:
        leaves_only = [lf for (lf, p) in leaves_in_order]
        node_id2avg_gamma = {}

        def init_dict(n):
            node_id2avg_gamma[id(n)] = None
            if not n.is_leaf:
                init_dict(n.left_child)
                init_dict(n.right_child)
        init_dict(model.root)

        def compute_leaf_dist_soft(node, x_):
            if node.is_leaf:
                return torch.ones(x_.size(0),1, device=device)
            else:
                if node.gating is not None:
                    gate_logit = node.gating(x_).squeeze(-1)
                else:
                    gate_logit = x_ @ node.w + node.b
                p_left = torch.sigmoid(gate_logit)
                left_ = compute_leaf_dist_soft(node.left_child, x_)
                right_= compute_leaf_dist_soft(node.right_child, x_)
                return torch.cat([left_*p_left.unsqueeze(-1), right_*(1-p_left).unsqueeze(-1)], dim=1)

        with torch.no_grad():
            dist_all = compute_leaf_dist_soft(model.root, x)  # (N, #leaves)

        for i, leaf_node in enumerate(leaves_only):
            coverage_i = dist_all[:, i]  # shape (N,)
            cov_sum = coverage_i.sum().item()
            if cov_sum<1e-12:
                node_id2avg_gamma[id(leaf_node)] = np.zeros(D, dtype=np.float32)
                continue

            with torch.no_grad():
                phi_vec = torch.sigmoid(leaf_node.feature_selector.logit_phi).cpu().numpy()
            avg_gamma = phi_vec
            node_id2avg_gamma[id(leaf_node)] = avg_gamma

        for (leaf, path) in leaves_in_order:
            avg_gamma_leaf = node_id2avg_gamma[id(leaf)]
            w_infl = compute_local_expert_weight_influence(leaf.local_expert).cpu().numpy()
            leaf_infl = avg_gamma_leaf * w_infl

            # coverage
            i_ = leaves_only.index(leaf)
            coverage_leaf = dist_all[:, i_].sum().item()/float(N)

            idx_sorted = np.argsort(leaf_infl)
            idx_bottom = idx_sorted[:bottom_k]
            idx_top    = idx_sorted[-top_k:]

            top_features_info = []
            bottom_features_info = []
            for i in reversed(idx_top):
                top_features_info.append((
                    feature_names[i],
                    float(leaf_infl[i]),
                    float(avg_gamma_leaf[i]),
                    float(w_infl[i])
                ))
            for i in idx_bottom:
                bottom_features_info.append((
                    feature_names[i],
                    float(leaf_infl[i]),
                    float(avg_gamma_leaf[i]),
                    float(w_infl[i])
                ))

            leaf_influence_data.append({
                "leaf_path": path,
                "coverage": coverage_leaf,
                "top_features": top_features_info,
                "bottom_features": bottom_features_info,
            })

    return leaf_influence_data



def inference_and_explain(
    args,
    model,
    test_loader,
    gating_mode='hard',     # 'hard' or 'soft'
    temperature=0.1,       
    top_k_leaf=5,
    bottom_k_leaf=5,
    global_importance_mode='both',  
    # 'none'/'phi_only'/'gamma_weight'/'both'
    top_k_global=5,
    bottom_k_global=5,
    leaf_influence_mode='gamma_weight'  
    # 'none' or 'gamma_weight'
):

    device = getattr(args,'device','cpu')
    model.eval()

    # 1)  Accuracy, MCC
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
    cm = confusion_matrix(labels_array, preds_array)
    print(f"\n[Confusion Matrix ({gating_mode})]")
    print(cm)
    print("\n[Classification Report]")
    print(classification_report(labels_array, preds_array, digits=4))
    test_mcc = matthews_corrcoef(labels_array, preds_array)
    print(f"MCC: {test_mcc:.4f}")

    # 2) Node-level routing stats
    X_cat = torch.cat(all_x, dim=0)
    Y_cat = torch.cat(all_y, dim=0)

    node_level_stats = []
    if gating_mode=='hard':
        routing_info = gather_routing_statistics_hard(model.root, X_cat, Y_cat, device, "root")
        node_level_stats = routing_info  
        print("\n=== Node-level Routing Statistics (Hard Gating) ===")
        for ri in routing_info:
            if ri['is_leaf']:
                acc_str = f"{ri['acc_leaf']:.4f}" if ri['acc_leaf'] is not None else "NA"
                mcc_str = f"{ri['mcc_leaf']:.4f}" if ri['mcc_leaf'] is not None else "NA"
                print(f"[Leaf] {ri['node_path']} | #samples={ri['n_samples']} "
                      f"| label0={ri['n_label0']}, label1={ri['n_label1']}, "
                      f"leaf_acc={acc_str}, leaf_mcc={mcc_str}")
            else:
                print(f"[Node] {ri['node_path']} | #samples={ri['n_samples']} "
                      f"| label0={ri['n_label0']}, label1={ri['n_label1']}")
    else:
        coverage_init = torch.ones(X_cat.size(0), device=device)
        routing_info_soft = gather_routing_statistics_soft(model.root, X_cat, Y_cat, coverage_init, device, "root")
        node_level_stats = routing_info_soft  
        print("\n=== Node-level Routing Statistics (Soft Gating) ===")
        for ri in routing_info_soft:
            if ri['is_leaf']:
                acc_str = f"{ri['acc_leaf']:.4f}" if ri['acc_leaf'] is not None else "NA"
                mcc_str = f"{ri['mcc_leaf']:.4f}" if ri['mcc_leaf'] is not None else "NA"
                print(f"[Leaf] {ri['node_path']} | coverage={ri['coverage']:.3f} "
                      f"| label0={ri['coverage_label0']:.3f}, label1={ri['coverage_label1']:.3f}, "
                      f"acc={acc_str}, mcc={mcc_str}")
            else:
                print(f"[Node] {ri['node_path']} | coverage={ri['coverage']:.3f} "
                      f"| label0={ri['coverage_label0']:.3f}, label1={ri['coverage_label1']:.3f}")

    # 3) Leaf별 phi log-odds Top/Bottom K
        
    if hasattr(test_loader.dataset, 'feature_names'):
        feature_names = test_loader.dataset.feature_names
    else:
        feature_names = [f"feat_{i}" for i in range(model.dim_input)]

    print_leaf_top_bottom_features(model.root, feature_names, top_k=top_k_leaf, bottom_k=bottom_k_leaf)
    leaf_features_data = gather_leaf_feature_selection_data(model.root,
                                                            feature_names,
                                                            top_k=top_k_leaf,
                                                            bottom_k=bottom_k_leaf)

    # 4) Global Importance

    global_importance_data = {}
    def print_global_top_bottom(global_imp, feat_names, top_k=5, bottom_k=5, title="Global Importance"):
        idx_sorted = np.argsort(global_imp)
        idx_bottom = idx_sorted[:bottom_k]
        idx_top    = idx_sorted[-top_k:]
        print(f"\n=== {title} (Top {top_k}, Bottom {bottom_k}) ===")
        print("[Top Features]")
        for i in reversed(idx_top):
            print(f"   {feat_names[i]} => {global_imp[i]:.6f}")
        print("[Bottom Features]")
        for i in idx_bottom:
            print(f"   {feat_names[i]} => {global_imp[i]:.6f}")

    if global_importance_mode != 'none':
        X_cat = X_cat.to(device)

        # (A) phi_only
        if global_importance_mode in ['phi_only','both']:
            if gating_mode=='hard':
                g_imp_a, cov_dict_a = compute_global_importance_hard(model, X_cat, device)
            else:
                g_imp_a, cov_dict_a = compute_global_importance_soft(model, X_cat, device, temperature=temperature)

            print("\n=== Leaf Coverage (phi_only) ===")
            for path, cov in cov_dict_a.items():
                print(f"   {path} => coverage={cov:.4f}")

            print_global_top_bottom(
                g_imp_a,
                feature_names,
                top_k=top_k_global,
                bottom_k=bottom_k_global,
                title="Global Importance by (coverage * phi)"
            )

            global_importance_data["phi_only"] = {
                "importance": g_imp_a,
                "coverage_dict": cov_dict_a
            }

        # (B) gamma_weight
        if global_importance_mode in ['gamma_weight','both']:
            if gating_mode=='hard':
                g_imp_b, cov_dict_b = compute_global_influence_hard(model, X_cat, device)
            else:
                g_imp_b, cov_dict_b = compute_global_influence_soft(model, X_cat, device, temperature=temperature)

            print("\n=== Leaf Coverage (gamma_weight) ===")
            for path, cov in cov_dict_b.items():
                print(f"   {path} => coverage={cov:.4f}")

            print_global_top_bottom(
                g_imp_b,
                feature_names,
                top_k=top_k_global,
                bottom_k=bottom_k_global,
                title="Global Influence by (coverage * (avg_gamma x MLP))"
            )

            global_importance_data["gamma_weight"] = {
                "importance": g_imp_b,
                "coverage_dict": cov_dict_b
            }

    # 5) Leaf-level (gamma x MLP) influence
    
    leaf_influence = {}
    if leaf_influence_mode=='gamma_weight':
        data_ = collect_leaf_influence_gamma_weight(
            model=model,
            x=X_cat,
            feature_names=feature_names,
            gating_mode=gating_mode,
            temperature=temperature,
            top_k=top_k_leaf,
            bottom_k=bottom_k_leaf,
            device=device
        )
        leaf_influence["gamma_weight"] = data_
    else:
        leaf_influence["none"] = None

    print(f"\n>>> Inference & Explain Done. (mode={gating_mode}, "
          f"global_importance={global_importance_mode}, leaf_influence={leaf_influence_mode})")

    return (
        acc,
        test_mcc,
        node_level_stats,         # list of dict (hard or soft)
        leaf_features_data,       # list of dict (phi log-odds)
        global_importance_data,   # dict { "phi_only":{...}, "gamma_weight":{...} }
        leaf_influence            # dict { "gamma_weight":[...], "none":None }
    )
