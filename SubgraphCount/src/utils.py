import math
from collections import defaultdict
import random

from numpy.random import laplace
import numpy as np
import networkx as nx
from networkx.algorithms.community import k_clique_communities

def C(n, m):
    n = int(n)
    m = int(m)
    res = 1
    for i in range(n-m+1, n+1):
        res *= i
    return res // math.factorial(m)



def compute_price(price, epsilon, b):
    return price * 2 / math.pi * math.atan(abs(epsilon) / b)


def perturb_mechanism(data, scale, deta=10**(-4)):
    return data + laplace(scale=scale) + scale * math.log(1/(2*deta))


def induced_subgraph(graph, node_num):
    nodes = graph.nodes
    sampled_nodes = random.sample(nodes, node_num)
    subgraph = nx.subgraph(graph, sampled_nodes)
    return subgraph


def k_cliques(graph, k):
    # node -> number of k-cliques
    numbers = {}
    for node in graph:
        numbers[node] = 0
    for community_nodes in list(k_clique_communities(graph, k)):
        for community_node in community_nodes:
            numbers[community_node] += 1
    return numbers

