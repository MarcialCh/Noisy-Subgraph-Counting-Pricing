import math
import sys
from heapq import heapify, heappush, heappop

import networkx as nx
from utils import C, compute_price, perturb_mechanism

from numpy.random import laplace


def upper_local_sensitivity_mapping_by_2(graph, epsilon_0, k, delta=10 ** (-4)):
    total_node_num = len(graph.nodes)
    h_0 = 100
    deta_parameter = delta / (2 * h_0 + 2)
    perturb_degree = {}
    upper_local_sensitivity = {}
    for node in graph.nodes:
        upper_local_sensitivity[node] = 0
    for node in graph.nodes:
        perturb_degree[node] = perturb_mechanism(graph.degree(node), scale=4 / epsilon_0, deta=deta_parameter)
    perturb_degree = dict(sorted(perturb_degree.items(), key=lambda d: d[0], reverse=True))
    index = 1
    for node, p_degree in perturb_degree.items():
        if 2 * index / epsilon_0 * math.log(1 / (2 * deta_parameter)) < perturb_degree[node]:
            index += 1
        else:
            break
    h = int(index / 2) + 1
    large_degree_nodes = perturb_degree.keys()
    second_set = set([list(large_degree_nodes)[i] for i in range(0, h + 1)])
    for node in graph:
        if node in second_set:
            neighbors = graph[node]
            common_neighbors_num = 0
            for neighbor in neighbors:
                # c(v_i)= max_j N(v_i) \cap N(v_j)
                common_neighbors_num = max(common_neighbors_num,
                                           len(list(nx.common_neighbors(graph, neighbor, node))))
            perturbed_common_neighbors_num = perturb_mechanism(common_neighbors_num, scale=2 * h / epsilon_0,
                                                               deta=deta_parameter)
            perturbed_common_neighbors_num = min(perturbed_common_neighbors_num, perturb_degree[node])
        else:
            perturbed_common_neighbors_num = perturb_degree[node]
        if k == 3:
            upper_local_sensitivity[node] = k * perturbed_common_neighbors_num
        else:
            upper_local_sensitivity[node] = k * C(perturbed_common_neighbors_num, k - 2)
    return upper_local_sensitivity


def total_num_from_privacy_budget(graph, numbers, epsilon, k, alpha=0.1):
    epsilon_0 = alpha * epsilon
    perturbed_total_num = 0
    upper_local_sensitivity = upper_local_sensitivity_mapping_by_2(graph, epsilon_0, k)
    for node, num in numbers.items():
        variance = 2 * pow(upper_local_sensitivity[node] / ((1-alpha)*epsilon), 2)
        perturbed_total_num += num + laplace(scale=math.sqrt(variance / 2))
    return perturbed_total_num


def total_price_from_variance(graph, variance, price_points, k, b, alpha=1):
    total_price = 0.0
    epsilon_0 = alpha / math.sqrt(variance / 2)
    upper_local_sensitivity = upper_local_sensitivity_mapping_by_2(graph, epsilon_0, k)
    for node in graph.nodes:
        epsilon = upper_local_sensitivity[node] / pow(variance / 2, 0.5)
        total_price += compute_price(price_points[node], epsilon + epsilon_0, b)
    return total_price


def min_max_price_from_variance(graph, variance, price_points, k, b, alpha=1):
    epsilon_0 = alpha / math.sqrt(variance / 2)
    max_value, min_value, mean_value_1, mean_value_2 = 0, 0, 0, 0
    heap = []
    heapify(heap)
    upper_local_sensitivity = upper_local_sensitivity_mapping_by_2(graph, epsilon_0, k)
    for node in graph.nodes:
        epsilon = upper_local_sensitivity[node] / pow(variance / 2, 0.5)
        price = compute_price(price_points[node], epsilon + epsilon_0, b)
        heappush(heap, price)
    size = len(heap)
    middle = int(size / 3)
    for i in range(size):
        if i == 0:
            min_value = heappop(heap)
        elif i == middle:
            mean_value_1 = heappop(heap)
        elif i == middle * 2:
            mean_value_2 = heappop(heap)
        elif i == size - 1:
            max_value = heappop(heap)
        else:
            heappop(heap)
    return min_value, mean_value_1, mean_value_2, max_value
