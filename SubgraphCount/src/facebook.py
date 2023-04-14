#!/usr/bin/env python
import networkx as nx
import numpy as np
import glob
import os, os.path
import baseline
import improve
import two_phase
from utils import k_cliques, induced_subgraph
import time

pathhack = os.path.dirname(os.path.realpath(__file__))

feat_file_name = "%s/feature_map.txt" % (pathhack,)
feature_index = {}  # numeric index to name
inverted_feature_index = {}  # name to numeric index
network = nx.Graph()
ego_nodes = []


def parse_featname_line(line):
    line = line[(line.find(' ')) + 1:]  # chop first field
    split = line.split(';')
    name = ';'.join(split[:-1])  # feature name
    index = int(split[-1].split(" ")[-1])  # feature index
    return index, name


def load_features():
    # may need to build the index first
    if not os.path.exists(feat_file_name):
        feat_index = {}
        # build the index from data/*.featnames files
        featname_files = glob.iglob("%s/data/*.featnames" % pathhack)
        for featname_file_name in featname_files:
            featname_file = open(featname_file_name, 'r')
            for line in featname_file:
                # example line:
                # 0 birthday;anonymized feature 376
                index, name = parse_featname_line(line)
                feat_index[index] = name
            featname_file.close()
        keys = feat_index.keys()
        keys.sort()
        out = open(feat_file_name, 'w')
        for key in keys:
            out.write("%d %s\n" % (key, feat_index[key]))
        out.close()

    # index built, read it in (even if we just built it by scanning)
    global feature_index
    global inverted_feature_index
    index_file = open(feat_file_name, 'r')
    for line in index_file:
        split = line.strip().split(' ')
        key = int(split[0])
        val = split[1]
        feature_index[key] = val
    index_file.close()

    for key in feature_index.keys():
        val = feature_index[key]
        inverted_feature_index[val] = key


def load_nodes():
    assert len(feature_index) > 0, "call load_features() first"
    global network
    global ego_nodes
    # get all the node ids by looking at the files
    ego_nodes = [int(x.split("\\")[-1].split('.')[0]) for x in glob.glob("%s\\data/*.featnames" % pathhack)]
    node_ids = ego_nodes

    # parse each node
    for node_id in node_ids:
        featname_file = open("%s/data/%d.featnames" % (pathhack, node_id), 'r')
        feat_file = open("%s/data/%d.feat" % (pathhack, node_id), 'r')
        egofeat_file = open("%s/data/%d.egofeat" % (pathhack, node_id), 'r')
        edge_file = open("%s/data/%d.edges" % (pathhack, node_id), 'r')

        # parse ego node
        network.add_node(node_id)
        # 0 1 0 0 0 ...
        ego_features = [int(x) for x in egofeat_file.readline().split(' ')]
        i = 0
        network.nodes[node_id]['features'] = np.zeros(len(feature_index))
        for line in featname_file:
            key, val = parse_featname_line(line)
            network.nodes[node_id]['features'][key] = ego_features[i] + 1
            i += 1

        # parse neighboring nodes
        for line in feat_file:
            featname_file.seek(0)
            split = [int(x) for x in line.split(' ')]
            node_id = split[0]
            features = split[1:]
            network.add_node(node_id)
            network.nodes[node_id]['features'] = np.zeros(len(feature_index))
            i = 0
            for line in featname_file:
                key, val = parse_featname_line(line)
                network.nodes[node_id]['features'][key] = features[i]
                i += 1

        featname_file.close()
        feat_file.close()
        egofeat_file.close()
        edge_file.close()


def load_edges():
    global network
    assert network.order() > 0, "call load_nodes() first"
    edge_file = open("%s/facebook_combined.txt" % pathhack, "r")
    for line in edge_file:
        # nodefrom nodeto
        split = [int(x) for x in line.split(" ")]
        node_from = split[0]
        node_to = split[1]
        network.add_edge(node_from, node_to)


def load_network():
    """
    Load the network.  After calling this function, facebook.network points to a networkx object for the facebook data.

    """
    load_features()
    load_nodes()
    load_edges()


def feature_matrix():
    n_nodes = network.number_of_nodes()
    n_features = len(feature_index)

    X = np.zeros((n_nodes, n_features))
    for i, node in enumerate(network.nodes()):
        X[i, :] = network.nodes[node]['features']

    return X


def universal_feature(feature_index):
    """
    Does every node have this feature?

    """
    return len([x for x in network.nodes() if network.nodes[x]['feautures'][feature_index] > 0]) // network.order() == 1


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


    Ftest(network.order(), 4039, "order")
    Ftest(network.size(), 88234, "size")
    Ftest(round(nx.average_clustering(network), 4), 0.6055, "clustering")
    k = 3
    total = 0
    n = len(network.nodes)

    price_points = {}
    for node in network.nodes:
        price_points[node] = 1

    '''
    # Exp-1: Accuracy influenced by privacy budget.
    for epsilon in range(-6, 4):
        if k == 3:
            subgraph_numbers = nx.triangles(network)
        else:
            subgraph_numbers = k_cliques(network, k)
        for subgraph_num in subgraph_numbers.values():
            total += subgraph_num
        #perturbed_answer = baseline.total_num_from_privacy_budget(subgraph_numbers, 10**epsilon, k)
        #perturbed_answer = improve.total_num_from_privacy_budget(network, subgraph_numbers, 10**epsilon, k)
        perturbed_answer = two_phase.total_num_from_privacy_budget(network, subgraph_numbers,10**epsilon, k)
        perturbed_answer = max(0, perturbed_answer)
        accuracy = 1 - abs(perturbed_answer - total) / abs(perturbed_answer + total)
        print("%s:accuracy %f" % (epsilon, accuracy))
    
    
    # Exp-2: Impact of payment on variance.
    for variance in range(1, 6):
        total_price = baseline.total_price_from_variance(network, 40 * variance, price_points, k, 10**2)
        #total_price = improve.total_price_from_variance(network, 40 * variance, price_points, k, 10 ** 2, 1000)
        #total_price = two_phase.total_price_from_variance(network, 40*variance,price_points, k, 10**2, 1000)
        print("%s:variance %f" % (variance, total_price / n))

  
    # Exp-3: micro-payment for data sellers. 
    for variance in range(1, 6):
        #min_price, mean_price_1, mean_price_2, max_price = improve.min_max_price_from_variance(network, 40*variance, price_points, k, 10**2, 1000)
        min_price, mean_price_1, mean_price_2, max_price = two_phase.min_max_price_from_variance(network,  40 * variance, price_points, k, 10**2, 1000)
        print("%s:variance min: %f mean_1:%f mean_2:%f max :%f " % (variance, min_price, mean_price_1, mean_price_2, max_price))
    
    
    for data_proportion in range(1, 11):
        subgraph_node_num = int(0.1 * data_proportion * n)
        subgraph = induced_subgraph(network, subgraph_node_num)
        time_start = time.perf_counter()
        #total_price = baseline.total_price_from_variance(subgraph, 40, price_points, k, 10**3)
        total_price = improve.total_price_from_variance(subgraph, 40, price_points, k, 10 ** 2, 1000)
        #total_price = two_phase.total_price_from_variance(subgraph, 40, price_points, k, 1)
        time_end = time.perf_counter()
        print("%s:data_proportion %f" % (data_proportion, time_end-time_start))
    '''
    for k_examine in range(3,11):
        subgraph_node_num = n
        subgraph = induced_subgraph(network, subgraph_node_num)
        time_start = time.perf_counter()
        total_price = baseline.total_price_from_variance(subgraph, 40, price_points, k_examine, 10**3)
        #total_price = improve.total_price_from_variance(subgraph, 40, price_points, k_examine, 10 ** 2, 1000)
        #total_price = two_phase.total_price_from_variance(subgraph, 40, price_points, k_examine, 1)
        time_end = time.perf_counter()
        print("%s:k %f" % (k, time_end - time_start))

