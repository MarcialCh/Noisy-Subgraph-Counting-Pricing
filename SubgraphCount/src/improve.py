import math
import sys
from heapq import heapify, heappush, heappop

import networkx as nx
from utils import C, compute_price, perturb_mechanism
import time
from numpy.random import laplace


def upper_local_sensitivity_mapping(graph, epsilon_0, k):
    total_node_num = len(graph.nodes)
    upper_local_sensitivity = {}
    for node in graph.nodes:
        upper_local_sensitivity[node] = 0
    if k == 3:
        impact_num = total_node_num
    else:
        impact_num = total_node_num * (C(total_node_num - 2, k - 2) - C(total_node_num - 3, k - 2))
    for node in graph:
        neighbors = graph[node]
        common_neighbors_num = 0
        for neighbor in neighbors:
            # c(v_i)= max_j N(v_i) \cap N(v_j)
            common_neighbors_num = max(common_neighbors_num,
                                       len(list(nx.common_neighbors(graph, neighbor, node))))
        perturbed_common_neighbors_num = perturb_mechanism(common_neighbors_num, scale=impact_num / epsilon_0)
        if k == 3:
            upper_local_sensitivity[node] = k * perturbed_common_neighbors_num
        else:
            upper_local_sensitivity[node] = k * C(perturbed_common_neighbors_num, k - 2)
    return upper_local_sensitivity


def total_num_from_privacy_budget(graph, numbers, epsilon, k, alpha=0.1):
    perturbed_total_num = 0.0
    # initial privacy budget for local sensitivity
    epsilon_0 = alpha * epsilon
    upper_local_sensitivity = upper_local_sensitivity_mapping(graph, epsilon_0, k)
    for node, num in numbers.items():
        variance = 2 * pow(upper_local_sensitivity[node] / ((1 - alpha) * epsilon), 2)
        perturbed_total_num += num + laplace(scale=math.sqrt(variance / 2))
    return perturbed_total_num


def total_price_from_variance(graph, variance, price_points, k, b, alpha=0.1):
    total_price = 0.0
    epsilon_0 = alpha / math.sqrt(variance / 2)
    upper_local_sensitivity = upper_local_sensitivity_mapping(graph, epsilon_0, k)
    for node in graph.nodes():
        epsilon = upper_local_sensitivity[node] / pow(variance / 2, 0.5)
        total_price += compute_price(price_points[node], epsilon + epsilon_0, b)
    return total_price


def min_max_price_from_variance(graph, variance, price_points, k, b, alpha=0.1):
    epsilon_0 = alpha / math.sqrt(variance / 2)
    upper_local_sensitivity = upper_local_sensitivity_mapping(graph, epsilon_0, k)
    max_value, min_value, mean_value_1, mean_value_2 = 0, 0, 0, 0
    heap = []
    heapify(heap)
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
