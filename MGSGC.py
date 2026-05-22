import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import math
import argparse
import numpy as np
from scipy.linalg import eigh
from scipy.stats import wasserstein_distance

import coarsening
import util
import networkx as nx
import parse
from classification import _normalizeLaplacian, _laplacian, laplacian
import classification
from laplacian import normalizeLaplacian
import os
import sys
import netlsd
import explog
import time
import networkx.algorithms.community as community

parser = argparse.ArgumentParser(description='Experiment for graph classification with coarse graphs')
util.assign_parser(parser)
args = parser.parse_args()
util.check_args(args)

explog.set_all_seeds(args.seed)

def calculate_laplacian_spectrum(graph):
    laplacian_matrix = nx.normalized_laplacian_matrix(graph).toarray()
    spectrum = eigh(laplacian_matrix, eigvals_only=True)
    return spectrum

def spectral_distance(spectrum1, spectrum2, bins=100):
    hist1, _ = np.histogram(spectrum1, bins=bins, range=(0, 2), density=True)
    hist2, _ = np.histogram(spectrum2, bins=bins, range=(0, 2), density=True)
    return wasserstein_distance(hist1, hist2)

def split_ball(graph, split_GB_list, spectrum_threshold=0.1, max_depth=3, current_depth=0):
    if len(graph) == 1 or current_depth >= max_depth:
        split_GB_list.append(graph)
        return

    laplacian_matrix = nx.normalized_laplacian_matrix(graph).toarray()
    eigvals, eigvecs = eigh(laplacian_matrix)
    fiedler_vector = eigvecs[:, 1]
    cluster_a = [node for i, node in enumerate(graph.nodes()) if fiedler_vector[i] < 0]
    cluster_b = [node for i, node in enumerate(graph.nodes()) if fiedler_vector[i] >= 0]

    graph_a = graph.subgraph(cluster_a)
    graph_b = graph.subgraph(cluster_b)

    if len(graph_a.edges()) == 0 or len(graph_b.edges()) == 0:
        split_GB_list.append(graph)
        return

    original_spectrum = calculate_laplacian_spectrum(graph)
    spectrum_a = calculate_laplacian_spectrum(graph_a)
    spectrum_b = calculate_laplacian_spectrum(graph_b)

    spectral_dist = spectral_distance(original_spectrum, np.concatenate([spectrum_a, spectrum_b]))

    if spectral_dist < spectrum_threshold:
        split_ball(graph_a, split_GB_list, spectrum_threshold, max_depth, current_depth + 1)
        split_ball(graph_b, split_GB_list, spectrum_threshold, max_depth, current_depth + 1)
    else:
        split_GB_list.append(graph)

import math
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

def init_GB_graph(graph):
    communities = list(greedy_modularity_communities(graph))
    
    init_GB_list = []
    for comm in communities[:int(math.sqrt(len(graph)))]:
        subgraph = graph.subgraph(comm).copy()
        init_GB_list.append(subgraph)
    
    return init_GB_list

def get_GB_graph(graph):
    init_GB_list = init_GB_graph(graph)
    GB_list = []
    for init_GB in init_GB_list:
        split_GB_list = []
        split_ball(init_GB, split_GB_list)
        GB_list.extend(split_GB_list)
    
    GB_graph = nx.Graph()
    if len(GB_list) == 1:
        return graph
    for i in range(len(GB_list)):
        GB_graph.add_node(i)
    for i in range(len(GB_list)):
        for j in range(i + 1, len(GB_list)):
            weight = 0
            for a in GB_list[i].nodes():
                for b in GB_list[j].nodes():
                    if graph.has_edge(a, b):
                        weight += 1
            if weight > 0:
                GB_graph.add_edge(i, j, weight=weight)
    return GB_graph

dir = 'dataset'
datasets = ['COX2', 'MSRC_9']

for dataset in datasets:
    am, labels = parse.parse_dataset(dir, dataset)
    G_list = []

    print(dataset)
    total_original_nodes = 0
    total_coarsened_nodes = 0

    num_samples = len(am)
    Y = labels
    current_timestamp = time.time()
    
    for i in range(num_samples):
        N = am[i].shape[0]
        graph = nx.from_numpy_matrix(am[i])
        total_original_nodes += N

        if nx.is_connected(graph):
            GB_graph = get_GB_graph(graph)
        else:
            connected_components = list(nx.connected_components(graph))
            connected_subgraphs = [graph.subgraph(component) for component in connected_components]
            GB_graph_list = []
            for connected_subgraph in connected_subgraphs:
                GB_graph = get_GB_graph(connected_subgraph)
                GB_graph_list.append(GB_graph)
            GB_graph = nx.Graph()
            for idx, subgraph in enumerate(GB_graph_list):
                mapping = {node: node + len(GB_graph.nodes()) for node in subgraph.nodes()}
                subgraph = nx.relabel_nodes(subgraph, mapping)
                GB_graph = nx.compose(GB_graph, subgraph)

        total_coarsened_nodes += len(GB_graph)

        GB_graph_am = nx.to_numpy_array(GB_graph)
        G = eigh(normalizeLaplacian(GB_graph_am), eigvals_only=True)
        G_list.append(G)

    coarsening_ratio = (total_coarsened_nodes / total_original_nodes) * 100

    best_acc = 0
    best_std = 0
    best_h = 0
    for j in range(1, 300):
        t = np.logspace(-2, 2, j)
        X = np.zeros((num_samples, j))
        for i in range(len(G_list)):
            X[i] = netlsd.heat(G_list[i], t, normalization="empty")
        acc, std = classification.KNN_classifier_nfold(X, Y, n=10, k=1)
        if acc > best_acc:
            best_acc = acc
            best_std = std
            best_h = j
    print(dataset, "Average Accuracy={:.2f}±{:.2f}  h={}".format(best_acc, best_std / np.sqrt(10), best_h))













