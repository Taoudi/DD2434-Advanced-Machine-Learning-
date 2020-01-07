""" This file is created as the solution template for question 2.3 in DD2434 - Assignment 2.

    Please keep the fixed parameters in the function templates as is (in 2_3.py file).
    However if you need, you can add parameters as default parameters.
    i.e.
    Function template: def calculate_likelihood(tree_topology, theta, beta):
    You can change it to: def calculate_likelihood(tree_topology, theta, beta, new_param_1=[], new_param_2=123):

    You can write helper functions however you want.

    If you want, you can use the class structures provided to you (Node, Tree and TreeMixture classes in Tree.py
    file), and modify them as needed. In addition to the sample files given to you, it is very important for you to
    test your algorithm with your own simulated data for various cases and analyse the results.

    For those who do not want to use the provided structures, we also saved the properties of trees in .txt and .npy
    format.

    Also, I am aware that the file names and their extensions are not well-formed, especially in Tree.py file
    (i.e example_tree_mixture.pkl_samples.txt). I wanted to keep the template codes as simple as possible.
    You can change the file names however you want (i.e tmm_1_samples.txt).

    For this assignment, we gave you three different trees (q_2_3_small_tree, q_2_3_medium_tree, q_2_3_large_tree).
    Each tree have 5 samples (whose inner nodes are masked with np.nan values).
    We want you to calculate the likelihoods of each given sample and report it.
"""
from collections import defaultdict

import numpy as np
from Tree import Tree
from Tree import Node


# Starting from root
def find_leaves(beta):
    leaves = []
    for node, val in enumerate(beta):
        if not np.isnan(val):
            leaves.append(node)
    return leaves


def find_children(node, topology, beta):
    children = []

    for index, parent in enumerate(topology):
        if parent == node:
            children.append(index)
    return children


def CPD(theta, node, cat, parent_cat=None):
    if parent_cat is None:
        return theta[node][cat]
    else:
        return theta[node][int(parent_cat)][int(cat)]


# Calculating s values for the tree and storing them
def s_root(tree_topology, theta, beta):
    prob = 0
    s_storage = defaultdict(dict)

    def S(u, j, children):
        if s_storage[u].get(j) is not None:
            return s_storage[u].get(j)
        if len(children) < 1:
            if beta.astype(int)[u] == j:
                s_storage[u][j] = 1
                return 1
            else:
                s_storage[u][j] = 0
                return 0
        #result = np.zeros(len(children))
        result = [0]*len(children)
        for child_nr, child in enumerate(children):
            for category in range(0, len(theta[0])):
                result[child_nr] += S(child, category, find_children(child, tree_topology, beta)) * CPD(theta, child,
                                                                                                        category, j)
        res = 1
        for i in range(len(children)):
            res *= result[i]
        s_storage[u][j] = res
        #res = np.prod(result)
        return res

    for i, th in enumerate(theta[0]):
        prob += S(0, i, find_children(0, tree_topology, beta)) * CPD(theta, 0, i)
    return s_storage


def find_sibling(u, topology):
    for node, parent in enumerate(topology):
        if np.isnan(parent) and np.isnan(topology[u]) and u != node:
            return node
        elif parent == topology[u] and u != node:
            return node
    return None


def calculate_likelihood(tree_topology, theta, beta):
    """
    This function calculates the likelihood of a sample of leaves.
    :param: tree_topology: A tree topology. Type: numpy array. Dimensions: (num_nodes, )
    :param: theta: CPD of the tree. Type: numpy array. Dimensions: (num_nodes, K)
    :param: beta: A list of node assignments. Type: numpy array. Dimensions: (num_nodes, )
                Note: Inner nodes are assigned to np.nan. The leaves have values in [K]
    :return: likelihood: The likelihood of beta. Type: float.

    You can change the function signature and add new parameters. Add them as parameters with some default values.
    i.e.
    Function template: def calculate_likelihood(tree_topology, theta, beta):
    You can change it to: def calculate_likelihood(tree_topology, theta, beta, new_param_1=[], new_param_2=123):
    """

    # TODO Add your code here
    print("-----------------------------------------")
    print("Calculating the likelihood...")

    s_dict = s_root(tree_topology, theta, beta)
    t_dict = defaultdict(dict)
    likelihood = 1

    # Calculating t dynamically
    def t(u, i, parent, sibling):
        if np.isnan(parent):  # If root
            return CPD(theta, u, i)  # * s_dict[u][i]
        if t_dict[u].get(i) is not None:  # If it has already been calculated
            return t_dict[u].get(i)

        parent = int(parent)
        result = 0
        for j in range(0, len(theta[0])):
            for k in range(0, len(theta[0])):
                result += CPD(theta, u, i, j) * CPD(theta, sibling, k, j) * s_dict[sibling].get(k) * t(parent, j,
                                                                                                       tree_topology[
                                                                                                           parent],
                                                                                                       find_sibling(
                                                                                                           parent,
                                                                                                           tree_topology))
        t_dict[u][i] = result
        return result

    for leaf, cat in enumerate(beta):
        if not np.isnan(cat):
            part_likelihood = t(leaf, cat, tree_topology[leaf],
                                find_sibling(leaf, tree_topology)) * s_dict[leaf].get(cat)
            print(part_likelihood)
            return part_likelihood


def main():
    print("Hello World!")
    print("This file is the solution template for question 2.3.")

    print("\n1. Load tree data from file and print it\n")

    filename = "data2_3/q2_3_large_tree.pkl"  # "data/q2_3_medium_tree.pkl", "data/q2_3_large_tree.pkl"
    t = Tree()
    t.load_tree(filename)
    t.print()

    print("\n2. Calculate likelihood of each FILTERED sample\n")
    # These filtered samples already available in the tree object.
    # Alternatively, if you want, you can load them from corresponding .txt or .npy files

    for sample_idx in range(t.num_samples):
        beta = t.filtered_samples[sample_idx]
        print("\n\tSample: ", sample_idx, "\tBeta: ", beta)
        sample_likelihood = calculate_likelihood(t.get_topology_array(), t.get_theta_array(), beta)
        print("\tLikelihood: ", sample_likelihood)


if __name__ == "__main__":
    main()
