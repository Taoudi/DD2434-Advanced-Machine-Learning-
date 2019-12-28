# Code segment taken from:
# https://stackoverflow.com/questions/39091191/wrong-output-in-implementing-kruskal-algorithm-in-python
# and fixed / modified for this exercise

parent = dict()
rank = dict()


def make_set(vertex):
    parent[vertex] = vertex
    rank[vertex] = 0


def find(vertex):
    if parent[vertex] != vertex:
        parent[vertex] = find(parent[vertex])
    return parent[vertex]


def union(vertex1, vertex2):
    root1 = find(vertex1)
    root2 = find(vertex2)
    if root1 != root2:
        if rank[root1] > rank[root2]:
            parent[root2] = root1
        else:
            parent[root1] = root2
        if rank[root1] == rank[root2]:
            rank[root2] += 1


def kruskal(graph):
    for vertex in graph['vertices']:
        make_set(vertex)
    minimum_spanning_tree = set()
    edges = list(graph['edges'])
    edges.sort(key=lambda tup: tup[2])

    for edge in edges:
        vertex1, vertex2, weight = edge
        if find(vertex1) != find(vertex2):
            union(vertex1, vertex2)
            minimum_spanning_tree.add(edge)
    return sorted(minimum_spanning_tree)


def maximum_spanning_tree(graph):
    """ This function is the modified version of kruskal function.
        Given a graph with weighted edges, this function returns the maximum spanning tree. """

    print("Running maximum spanning tree algorithm...")

    for vertex in graph['vertices']:
        make_set(vertex)
    mst = set()
    edges = list(graph['edges'])
    edges.sort(key=lambda tup: tup[2])
    edges = edges[::-1]
    # print(edges)

    flag = False
    for edge in edges:
        vertex1, vertex2, weight = edge
        if find(vertex1) != find(vertex2):
            union(vertex1, vertex2)
            if len(list(mst)) < len(graph['vertices']) - 1:
                mst.add(edge)
                # if weight == 0:
                # print("MST Warning. 0 weighted edge is used. Vertices: ", vertex1, vertex2)
            else:
                flag = True
    if flag:
        print("MST. Number of edges reached the limit!")
    return sorted(mst)


def main():
    print("Hello World!")
    print("This file demonstrates the usage of the functions.")
    print("The codes are taken from "
          "https://stackoverflow.com/questions/39091191/wrong-output-in-implementing-kruskal-algorithm-in-python "
          "and fixed/modified.")

    print("\nCreate a graph.")
    graph = {
        'vertices': [0, 1, 2, 3, 4, 5],
        'edges': {(0, 3, 5), (3, 5, 2), (5, 4, 10), (4, 1, 3), (1, 0, 8), (0, 2, 1), (2, 3, 6), (2, 5, 4), (2, 4, 9),
                  (2, 1, 7)}
    }

    print("\nRun Kruskal's algorithm.")

    min_spanning_tree = kruskal(graph)
    print("Minimum spanning tree: \n", min_spanning_tree)

    print("\nRun maximum spanning tree algorithm.")

    max_spanning_tree = maximum_spanning_tree(graph)
    print("Maximum spanning tree: \n", max_spanning_tree)


if __name__ == "__main__":
    main()