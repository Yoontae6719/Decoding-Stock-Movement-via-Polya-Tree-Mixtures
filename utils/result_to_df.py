import pandas as pd

def leaf_selection_to_dataframe(leaf_selection_list):
    rows = []
    for leaf_sel in leaf_selection_list:
        leaf_path = leaf_sel['leaf_path']
        
        top_feats = leaf_sel.get('top_features', [])
        for rank_i, (feat_name, score) in enumerate(top_feats, start=1):
            rows.append({
                "leaf_path": leaf_path,
                "rank": rank_i,
                "feature_name": feat_name,
                "score": score,
                "type": "top"
            })
        
        bottom_feats = leaf_sel.get('bottom_features', [])
        for rank_i, (feat_name, score) in enumerate(bottom_feats, start=1):
            rows.append({
                "leaf_path": leaf_path,
                "rank": rank_i,
                "feature_name": feat_name,
                "score": score,
                "type": "bottom"
            })
    
    df = pd.DataFrame(rows)
    return df


def leaf_influence_to_dataframe(leaf_influence_list):

    rows = []
    for leaf_info in leaf_influence_list:
        leaf_path = leaf_info['leaf_path']
        coverage = leaf_info['coverage']

        # 1) top_features
        top_feats = leaf_info.get('top_features', [])
        for rank_i, (feat_name, leaf_infl, gamma_val, weight_val) in enumerate(top_feats, start=1):
            rows.append({
                'leaf_path': leaf_path,
                'coverage': coverage,
                'rank': rank_i,
                'feature_name': feat_name,
                'leaf_infl': leaf_infl,
                'gamma': gamma_val,
                'weight': weight_val,
                'type': 'top'
            })

        # 2) bottom_features
        bottom_feats = leaf_info.get('bottom_features', [])
        for rank_i, (feat_name, leaf_infl, gamma_val, weight_val) in enumerate(bottom_feats, start=1):
            rows.append({
                'leaf_path': leaf_path,
                'coverage': coverage,
                'rank': rank_i,
                'feature_name': feat_name,
                'leaf_infl': leaf_infl,
                'gamma': gamma_val,
                'weight': weight_val,
                'type': 'bottom'
            })

    df = pd.DataFrame(rows)
    return df

def node_level_stats_to_dataframe(node_level_stats):

    rows = []
    for stat in node_level_stats:
        row = {
            'node_path': stat['node_path'],
            'coverage': stat.get('coverage', None),
            'coverage_label0': stat.get('coverage_label0', None),
            'coverage_label1': stat.get('coverage_label1', None),
            'acc_leaf': stat.get('acc_leaf', None),
            'mcc_leaf': stat.get('mcc_leaf', None),
            'is_leaf': stat.get('is_leaf', None)
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df



def global_importance_to_dataframes(global_importance_dict, feature_names=None):

    coverage_rows = []
    importance_rows = []

    for imp_type, data_dict in global_importance_dict.items():
        coverage_dict = data_dict.get('coverage_dict', {})
        imp_arr = data_dict.get('importance', None)

        # (A) coverage_dict -> DataFrame
        for leaf_path, cov_val in coverage_dict.items():
            coverage_rows.append({
                'type': imp_type,
                'leaf_path': leaf_path,
                'coverage': cov_val
            })

        # (B) importance -> DataFrame
        if imp_arr is not None:
            # 길이 D = feature 개수
            if feature_names is not None and len(feature_names) == len(imp_arr):
                for f_name, val in zip(feature_names, imp_arr):
                    importance_rows.append({
                        'type': imp_type,
                        'feature_name': f_name,
                        'importance': float(val)
                    })
            else:
                # feature_names 없는 경우 -> feature_idx 로만 저장
                for i, val in enumerate(imp_arr):
                    importance_rows.append({
                        'type': imp_type,
                        'feature_idx': i,
                        'importance': float(val)
                    })

    df_coverage = pd.DataFrame(coverage_rows)
    df_importance = pd.DataFrame(importance_rows)
    return df_coverage, df_importance

