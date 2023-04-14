import sys

import networkx as nx
import numpy as np
import glob
import os, os.path
import math

from heapq import heappop, heappush, heapify

def compute_price(price, epsilon, b):
    return price * 2 / math.pi * math.atan(abs(epsilon) / b)


def add_laplace_noise(beta):
    u1 = np.random.random()
    u2 = np.random.random()
    if u1 <= 0.5:
        n_value = -beta * np.log(1. - u2)
    else:
        n_value = beta * np.log(u2)
    return n_value


def baselineE(triangle_numbers, query, epsilon):
    perturbed_total_num = 0.0
    total_node = len(triangle_numbers)
    for i, triangle_num in enumerate(triangle_numbers.values()):
        DS = 3 * abs(query[i]) * (total_node - 1)
        variance = 2 * pow(DS / epsilon, 2)
        perturbed_total_num += triangle_num + add_laplace_noise(math.sqrt(variance / 2))
    return perturbed_total_num


def baselineV(triangle_numbers, query, price_points, variance):
    total_price = 0.0
    total_node = len(triangle_numbers)
    for i, triangle_num in enumerate(triangle_numbers.values()):
        DS = 3 * abs(query[i]) * (total_node - 1)
        epsilon = DS / pow(variance / 2, 0.5)
        #total_price += compute_price(price_points[i], epsilon, pow(10, 3))
        total_price += compute_price(price_points[i], epsilon, pow(10, 3.5))
    return total_price


def baselineM(triangle_numbers, query, price_points, variance):
    total_node = len(triangle_numbers)
    for i, triangle_num in enumerate(triangle_numbers.values()):
        DS = 3 * abs(query[i]) * (total_node - 1)
        epsilon = DS / pow(variance / 2, 0.5)
        return epsilon


def improvedE(graph, triangle_numbers, query, epsilon):
    perturbed_total_num = 0.0
    total_node = len(triangle_numbers)
    node_index_id = {}
    i = 0
    for node in graph.nodes:
        node_index_id[i] = node
        i += 1
    for i, triangle_num in enumerate(triangle_numbers.values()):
        LS = 0
        neighbors = graph[node_index_id[i]]
        for neighbor in neighbors:
            LS = max(len(set(neighbors).intersection(set(graph[neighbor]))), LS)
        epsilon_0 = 0.25 * epsilon
        LS_final = LS + (total_node / epsilon_0) + total_node / epsilon_0 * math.log(10, total_node / (2 * pow(10, -4)))
        DS = 3*abs(query[i]) * LS_final * total_node
        variance = 2 * pow(DS / epsilon, 2)
        # print("variance:%f" % variance)
        perturbed_total_num += triangle_num + add_laplace_noise(math.sqrt(variance / 2))
    return perturbed_total_num


def improvedV(graph, triangle_numbers, query, price_points, variance):
    total_price = 0.0
    total_node = len(triangle_numbers)
    node_index_id = {}
    i = 0
    for node in graph.nodes:
        node_index_id[i] = node
        i += 1
    for i, triangle_num in enumerate(triangle_numbers.values()):
        LS = 0
        neighbors = graph[node_index_id[i]]
        for neighbor in neighbors:
            LS = max(len(set(neighbors).intersection(set(graph[neighbor]))), LS)
        epsilon_0 = 1
        LS_final = LS + (total_node / epsilon_0) + total_node / epsilon_0 * math.log(10, total_node / (2 * pow(10, -4)))
        DS = abs(query[i]) * LS_final * total_node
        epsilon = 3*DS / pow(variance / 2, 0.5)
        #total_price += compute_price(price_points[i], epsilon, 2 * pow(10, 6))
        total_price += compute_price(price_points[i], epsilon, 2 * pow(10, 7))
    return total_price


def improvedM(graph, triangle_numbers, query, price_points, variance):
    max_value = 0
    min_value = sys.maxsize
    total_node = len(triangle_numbers)
    node_index_id = {}
    i = 0
    for node in graph.nodes:
        node_index_id[i] = node
        i += 1
    for i, triangle_num in enumerate(triangle_numbers.values()):
        LS = 0
        neighbors = graph[node_index_id[i]]
        for neighbor in neighbors:
            LS = max(len(set(neighbors).intersection(set(graph[neighbor]))), LS)
        epsilon_0 = 1
        LS_final = LS + (total_node / epsilon_0) + total_node / epsilon_0 * math.log(10, total_node / (2 * pow(10, -4)))
        DS = abs(query[i]) * LS_final * total_node
        epsilon = 3*DS / pow(variance / 2, 0.5)
        #price = compute_price(price_points[i], epsilon, 2 * pow(10, 6))
        price = compute_price(price_points[i], epsilon, 2 * pow(10, 7))
        min_value = min(min_value, price)
        max_value = max(max_value, price)
    return min_value, max_value


def two_phase(graph, triangle_numbers, query, epsilon):
    epsilon_0 = 0.25 * epsilon
    h_0 = 100
    deta = pow(10, -4)
    perturb_degree = {}
    perturbed_total_num = 0.0
    node_index_id = {}
    i=0
    for node in graph.nodes:
        node_index_id[i] = node
        i += 1
    for node in graph.nodes:
        perturb_degree[node] = graph.degree(node) + add_laplace_noise(4 / epsilon_0) + 4 / epsilon_0 * math.log(10, (
                    h_0 + 1) / deta)
    perturb_degree = dict(sorted(perturb_degree.items(), key=lambda d: d[0], reverse=True))
    index = 1
    for i, p_degree in enumerate(perturb_degree):
        if 2 * index / epsilon_0 * math.log(10, (h_0 + 1) / deta) >= perturb_degree[node_index_id[i]]:
            index = i + 1
            break
    h = int(index / 2)
    nodes_index = perturb_degree.keys()
    second_set = set([list(nodes_index)[i] for i in range(0, h + 1)])
    for i, triangle_num in enumerate(triangle_numbers.values()):
        if i in second_set:
            LS = 0
            neighbors = graph[node_index_id[i]]
            for neighbor in neighbors:
                LS = max(len(set(neighbors).intersection(set(graph[neighbor]))), LS)
            LS += add_laplace_noise(2 * h / epsilon_0) + 2 * h / epsilon_0 * math.log(10, (h_0 + 1) / deta)
            LS_final = min(LS, perturb_degree[node_index_id[i]])
        else:
            LS_final = perturb_degree[node_index_id[i]]
        DS = 3 * abs(query[i]) * LS_final
        variance = 2 * pow(DS / epsilon, 2)
        # print("variance:%f" % variance)
        perturbed_total_num += triangle_num + add_laplace_noise(math.sqrt(variance / 2))
    return perturbed_total_num


def two_phaseV(graph, triangle_numbers, query, price_points, variance):
    total_price = 0.0
    epsilon_0 = 1
    h_0 = 100
    deta = pow(10, -4)
    perturb_degree = {}
    perturbed_total_num = 0.0
    node_index_id = {}
    i = 0
    for node in graph.nodes:
        node_index_id[i] = node
        i += 1
    for node in graph.nodes:
        perturb_degree[node] = graph.degree(node) + add_laplace_noise(4 / epsilon_0) + 4 / epsilon_0 * math.log(10, (
                    h_0 + 1) / deta)
    perturb_degree = dict(sorted(perturb_degree.items(), key=lambda d: d[0], reverse=True))
    index = 1
    for i, p_degree in enumerate(perturb_degree):
        if 2 * index / epsilon_0 * math.log(10, (h_0 + 1) / deta) >= perturb_degree[node_index_id[i]]:
            index = i + 1
            break
    h = int(index / 2)
    nodes_index = perturb_degree.keys()
    second_set = set([list(nodes_index)[i] for i in range(0, h + 1)])
    for i, triangle_num in enumerate(triangle_numbers.values()):
        if i in second_set:
            LS = 0
            neighbors = graph[i]
            for neighbor in neighbors:
                LS = max(len(set(neighbors).intersection(set(graph[neighbor]))), LS)
            LS += add_laplace_noise(2 * h / epsilon_0) + 2 * h / epsilon_0 * math.log(10, (h_0 + 1) / deta)
            LS_final = min(LS, perturb_degree[node_index_id[i]])
        else:
            LS_final = perturb_degree[node_index_id[i]]
        DS = 3 * abs(query[i]) * LS_final
        epsilon = DS / pow(variance / 2, 0.5)
        total_price += compute_price(price_points[i], epsilon, 10)
    return total_price


def two_phaseM(graph, triangle_numbers, query, price_points, variance):
    epsilon_0 = 1
    h_0 = 100
    deta = pow(10, -4)
    perturb_degree = {}
    node_index_id = {}
    i = 0
    for node in graph.nodes:
        node_index_id[i] = node
        i += 1
    max_value, min_value, mean_value_1, mean_value_2 = 0,0,0,0
    heap = []
    heapify(heap)
    for node in graph.nodes:
        perturb_degree[node] = graph.degree(node) + add_laplace_noise(4 / epsilon_0) + 4 / epsilon_0 * math.log(10, (
                    h_0 + 1) / deta)
    perturb_degree = dict(sorted(perturb_degree.items(), key=lambda d: d[0], reverse=True))
    index = 1
    for i, p_degree in enumerate(perturb_degree):
        if 2 * index / epsilon_0 * math.log(10, (h_0 + 1) / deta) >= perturb_degree[node_index_id[i]]:
            index = i + 1
            break
    h = int(index / 2)
    nodes_index = perturb_degree.keys()
    second_set = set([list(nodes_index)[i] for i in range(0, h + 1)])
    j = 0
    for i, triangle_num in enumerate(triangle_numbers.values()):
        if i in enumerate(second_set):
            LS = 0
            neighbors = graph[i]
            for neighbor in neighbors:
                LS = max(len(set(neighbors).intersection(set(graph[neighbor]))), LS)
            LS += add_laplace_noise(2 * h / epsilon_0) + 2 * h / epsilon_0 * math.log(10, (h_0 + 1) / deta)
            LS_final = min(LS, perturb_degree[node_index_id[i]])
        else:
            LS_final = perturb_degree[node_index_id[i]]
        DS = 3 * abs(query[i]) * LS_final
        epsilon = DS / pow(variance / 2, 0.5)
        price = compute_price(price_points[i], epsilon, 10)
        heappush(heap,price)
    size = len(heap)
    middle = int(size/3)
    for i in range(size):
        if i == 0:
            min_value = heappop(heap)
        elif i == middle:
            mean_value_1 = heappop(heap)
        elif i == middle*2:
            mean_value_2 = heappop(heap)
        elif i == size - 1:
            max_value = heappop(heap)
        else:
            heappop(heap)
    return min_value, mean_value_1,mean_value_2, max_value

if __name__ == '__main__':
    print(add_laplace_noise(10))