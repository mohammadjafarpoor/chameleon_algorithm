from graphtools import *
import itertools
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

def internal_interconnectivity(graph, cluster):
    return np.sum(bisection_weights(graph, cluster))


def relative_interconnectivity(graph, cluster_i, cluster_j):
    edges = connecting_edges((cluster_i, cluster_j), graph)
    EC = np.sum(get_weights(graph, edges))
    ECci, ECcj = internal_interconnectivity(
        graph, cluster_i), internal_interconnectivity(graph, cluster_j)
    return EC / ((ECci + ECcj) / 2.0)


def internal_closeness(graph, cluster):
    cluster = graph.subgraph(cluster)
    edges = cluster.edges()
    weights = get_weights(cluster, edges)
    return np.sum(weights)


def relative_closeness(graph, cluster_i, cluster_j):
    edges = connecting_edges((cluster_i, cluster_j), graph)
    if not edges:
        return 0.0
    else:
        SEC = np.mean(get_weights(graph, edges))
    Ci, Cj = internal_closeness(
        graph, cluster_i), internal_closeness(graph, cluster_j)
    SECci, SECcj = np.mean(bisection_weights(graph, cluster_i)), np.mean(
        bisection_weights(graph, cluster_j))
    return SEC / ((Ci / (Ci + Cj) * SECci) + (Cj / (Ci + Cj) * SECcj))


def merge_score(g, ci, cj, a):
    return relative_interconnectivity(
        g, ci, cj) * np.power(relative_closeness(g, ci, cj), a)


def merge_best(graph, df, a, k, verbose=False):
    clusters = np.unique(df['cluster'])
    max_score = 0
    ci, cj = -1, -1
    if len(clusters) <= k:
        return False

    for combination in itertools.combinations(clusters, 2):
        i, j = combination
        if i != j:
            if verbose:
                print("Checking c%d c%d" % (i, j))
            gi = get_cluster(graph, [i])
            gj = get_cluster(graph, [j])
            edges = connecting_edges(
                (gi, gj), graph)
            if not edges:
                continue
            ms = merge_score(graph, gi, gj, a)
            if verbose:
                print("Merge score: %f" % (ms))
            if ms > max_score:
                if verbose:
                    print("Better than: %f" % (max_score))
                max_score = ms
                ci, cj = i, j

    if max_score > 0:
        if verbose:
            print("Merging c%d and c%d" % (ci, cj))
        df.loc[df['cluster'] == cj, 'cluster'] = ci
        for i, p in enumerate(graph.nodes()):
            if graph.nodes[p]['cluster'] == cj:
                graph.nodes[p]['cluster'] = ci
    return max_score > 0


def plot2d_graph(graph):
    pos = nx.get_node_attributes(graph, 'pos')
    # Manually choose colors
    cluster_colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'brown', 'cyan', 'pink']
    # Assign colors based on the cluster index
    c = [cluster_colors[i % len(cluster_colors)]
         for i in nx.get_node_attributes(graph, 'cluster').values()]
    fig, ax = plt.subplots()
    if c:  # is set
        nx.draw(graph, pos, node_color=c, node_size=0.25)
    else:
        nx.draw(graph, pos, node_size=0.25)
    plt.show(block=False)

def plot2d_data(df):
    if (len(df.columns) > 3):
        print("Plot Waring: more than 2-Dimensions!")
    df.plot(kind='scatter', c=df['cluster'], cmap='gist_rainbow', x=0, y=1)
    plt.show(block=False)

def cluster(df, k, knn=10, m=30, alpha=2.0, verbose=False, plot_graph=False, plot_data=False):
    graph = knn_graph(df, knn, verbose=verbose)
    graph = pre_part_graph(graph, m, df, verbose=verbose)
    iterm = tqdm(enumerate(range(m - k)), total=m-k)
    for i in iterm:
        merge_best(graph, df, alpha, k, verbose)
    res = rebuild_labels(df)
    if plot_graph:
      plot2d_graph(graph)
    if plot_data:
      plot2d_data(res)
    return res

def rebuild_labels(df):
    ans = df.copy()
    clusters = list(pd.DataFrame(df['cluster'].value_counts()).index)
    c = 1
    for i in clusters:
        ans.loc[df['cluster'] == i, 'cluster'] = c
        c = c + 1
    return ans
