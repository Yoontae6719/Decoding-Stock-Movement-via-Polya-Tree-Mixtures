import networkx as nx
import matplotlib.pyplot as plt

def visualize_node_level_stats_tree(
    node_level_stats,
    figsize=(10, 8),
    node_size=2000,
    font_size=10,
    edge_color='gray',
    arrowstyle='-|>',
    arrowsize=15,
    node_color_internal='#CCE5FF',  # 내부 노드 색
    node_color_leaf='#FFD1A9',      # leaf 노드 색
    show_plot=True,
    save_path=None
):

    path2info = { d['node_path']: d for d in node_level_stats }

    G = nx.DiGraph()

    for node_path, info in path2info.items():
        G.add_node(node_path)

    for node_path, info in path2info.items():
        if not info['is_leaf']:
            left_path = node_path + '_L'
            right_path= node_path + '_R'
            if left_path in path2info:
                G.add_edge(node_path, left_path)
            if right_path in path2info:
                G.add_edge(node_path, right_path)

    def make_label(info):
        lines = []
        node_path = info['node_path']
        lines.append(node_path)

        cov = info.get('coverage', None)
        if cov is not None:
            if isinstance(cov, float):
                lines.append(f"coverage={cov:.1f}")
            else:
                lines.append(f"coverage={cov}")

        # label0, label1
        c0 = info.get('coverage_label0', None)
        c1 = info.get('coverage_label1', None)
        if c0 is not None or c1 is not None:
            s0 = f"{c0:.1f}" if isinstance(c0, float) else f"{c0}"
            s1 = f"{c1:.1f}" if isinstance(c1, float) else f"{c1}"
            lines.append(f"label0={s0}, label1={s1}")

        # acc_leaf
        acc = info.get('acc_leaf', None)
        if acc is not None:
            lines.append(f"acc={acc:.3f}")

        # mcc_leaf
        mcc = info.get('mcc_leaf', None)
        if mcc is not None:
            lines.append(f"mcc={mcc:.3f}")

        return "\n".join(lines)

    labels = {}
    for node_path, info in path2info.items():
        labels[node_path] = make_label(info)

    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=TB')
    except:
        pos = nx.spring_layout(G)

    node_colors = []
    for node_path in G.nodes():
        info = path2info[node_path]
        if info['is_leaf']:
            node_colors.append(node_color_leaf)
        else:
            node_colors.append(node_color_internal)

    plt.figure(figsize=figsize)
    nx.draw_networkx_edges(G, pos, arrowstyle=arrowstyle, arrowsize=arrowsize, edge_color=edge_color)
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=font_size)
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[INFO] Tree plot saved to {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close()
