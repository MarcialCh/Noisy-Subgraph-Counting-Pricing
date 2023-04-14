#!/usr/bin/env python
import networkx as nx
import numpy as np
import glob
import os, os.path
import baseline, improve, two_phase
from utils import k_cliques, induced_subgraph
import time
pathhack = os.path.dirname(os.path.realpath(__file__))

network = nx.Graph()
ego_nodes = []







def load_edges():
    global network
    edge_file = open("%s/hepPh.txt" % pathhack, "r")
    for line in edge_file:
        # nodefrom nodeto
        split = [int(x) for x in line.split("\t")]
        node_from = split[0]
        node_to = split[1]
        network.add_node(node_from)
        network.add_node(node_to)
        network.add_edge(node_from, node_to)


def load_network():
    """
    Load the network.  After calling this function, facebook.network points to a networkx object for the facebook data.

    """
    load_edges()



if __name__ == '__main__':
    print("Running tests.")
    print("Loading network...")
    load_network()
    print("done.")

    failures = 0


    def Ftest(actual, expected, test_name):
        global failures  # lol python scope
        try:
            print("testing %s..." % (test_name,))
            assert actual == expected, "%s failed (%s != %s)!" % (test_name, actual, expected)
            print("%s passed (%s == %s)." % (test_name, actual, expected))
        except AssertionError as e:
            print(e)
            failures += 1


    Ftest(network.order(), 12008, "order")
    Ftest(network.size(), 118521, "size")
    Ftest(round(nx.average_clustering(network), 4), 0.6115, "clustering")
    k = 3
    total = 0
    price_points = {}
    n = len(network.nodes)
    for node in network.nodes:
        price_points[node] = 1
    d = dict(nx.degree(network))
    print("average degree:", sum(d.values()) / len(network.nodes))
    '''
    if k == 3:
        subgraph_numbers = nx.triangles(network)
    else:
        subgraph_numbers = k_cliques(network, k)
    for subgraph_num in subgraph_numbers.values():
        total += subgraph_num
    # Exp-1: Accuracy influenced by privacy budget.
    for epsilon in range(-6, 4):

        #perturbed_answer = baseline.total_num_from_privacy_budget(subgraph_numbers, 10**epsilon, k)
        #perturbed_answer = improve.total_num_from_privacy_budget(network, subgraph_numbers, 10**epsilon, k)
        perturbed_answer = two_phase.total_num_from_privacy_budget(network, subgraph_numbers,10**epsilon, k)
        perturbed_answer = max(0, perturbed_answer)
        accuracy = 1 - abs(perturbed_answer - total) / abs(perturbed_answer + total)
        print("%s:accuracy %f" % (epsilon, accuracy))
    '''

    # Exp-2: Impact of payment on variance.
    for variance in range(1, 6):
        #total_price = baseline.total_price_from_variance(network, 20 * variance, price_points, k, 10**2.5)
        total_price = improve.total_price_from_variance(network, 20 * variance, price_points, k, 10 ** 2.5, 1000)
        #total_price = two_phase.total_price_from_variance(network,20*variance,price_points,k,10 ** 2.5, 1000)
        print("%s:variance %f" % (variance, total_price / n))
    '''
    
    # Exp-3: micro-payment for data sellers.
    for variance in range(1, 6):
        #min_price, mean_price_1, mean_price_2, max_price = improve.min_max_price_from_variance(network, 20*variance, price_points, k, 10**2.5, 1000)
        min_price, mean_price_1, mean_price_2, max_price = two_phase.min_max_price_from_variance(network, 20 * variance, price_points, k, 10 ** 2.5, 1000)
        print("%s:variance min: %f mean_1:%f mean_2:%f max :%f " % (variance, min_price, mean_price_1, mean_price_2, max_price))
   
    for data_proportion in range(1, 11):
        subgraph_node_num = int(0.1 * data_proportion * n)
        subgraph = induced_subgraph(network, subgraph_node_num)
        sub_subgraph_numbers = nx.triangles(subgraph)
        time_start = time.perf_counter()
        total_price = baseline.total_price_from_variance(sub_subgraph_numbers, 20, price_points, k, 10**3)
        #total_price = improve.total_price_from_variance(subgraph, sub_subgraph_numbers, 20, price_points, k, 10 ** 2, 1000)
        #total_price = two_phase.total_price_from_variance(subgraph, sub_subgraph_numbers, 20, price_points, k, 1)
        time_end = time.perf_counter()
        print("%s:data_proportion %f" % (data_proportion, time_end-time_start))
        
    for data_proportion in range(1, 11):
        subgraph_node_num = int(0.1 * data_proportion * n)
        subgraph = induced_subgraph(network, subgraph_node_num)
        time_start = time.perf_counter()
        total_price = baseline.total_price_from_variance(subgraph, 20, price_points, k, 10 ** 3)
        #total_price = improve.total_price_from_variance(subgraph, 20, price_points, k, 10 ** 2, 1000)
        #total_price = two_phase.total_price_from_variance(subgraph, 20, price_points, k, 1)
        time_end = time.perf_counter()
        print("%s:data_proportion %f" % (data_proportion, time_end - time_start))
    

    for k_examine in range(3,11):
        subgraph_node_num = n
        subgraph = induced_subgraph(network, subgraph_node_num)
        time_start = time.perf_counter()
        #total_price = baseline.total_price_from_variance(subgraph, 20, price_points, k_examine, 10**3)
        #total_price = improve.total_price_from_variance(subgraph, 20, price_points, k_examine, 10 ** 2, 1000)
        #total_price = two_phase.total_price_from_variance(subgraph, 20, price_points, k_examine, 1)
        time_end = time.perf_counter()
        print("%s:k %f" % (k, time_end - time_start))
    '''

