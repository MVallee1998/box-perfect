import numpy as np
import json
import TE
import tqdm
degree = 8


import networkx as nx

def graph_with_cliques(arrays):
    """
    Creates a graph with cliques corresponding to an input array containing arrays of integers.

    Args:
        arrays: A list of arrays of integers.

    Returns:
        The corresponding graph.
    """
    G = Graph()
    for array in arrays:
        for i in range(len(array)):
            for j in range(i+1, len(array)):
                G.add_edge(array[i], array[j])
    return G


def getPolyhedronFromCliques(cliqueOfg,n):
	trivialConstraintsGe0 = np.array([np.zeros(n +1) for i in range(0,n)])
	c = 1
	for j in trivialConstraintsGe0:
		j[c]=1
		c = c+1
	cliqueConstraints = np.zeros((len(cliqueOfg),n+1))
	c=0
	for K in cliqueOfg:
		cliqueConstraints[c][0] = 1
		cliqueConstraints[c][K+1] = -1
		c=c+1
	constraintvectors = np.concatenate((trivialConstraintsGe0, cliqueConstraints))
	return Polyhedron(ieqs = constraintvectors,base_ring=QQ)
	
	
def read_file(filename):
    with open(filename, 'rb') as f:
        data = f.readlines()
    data = [x.strip() for x in data]
    return data



clique_path = "data_cliques/all_cliques_%d.txt" % degree
g6_path = "data_g6/perfect%d.g6" % degree


g6_data = [g6_byte.decode("utf-8") for g6_byte in read_file(g6_path)]
cliques_data = [json.loads(clique_bytes) for clique_bytes in read_file(clique_path)]
result = []
N = len(cliques_data)
for k in tqdm.tqdm(range(N)):
	cliques = [np.array(clique) for clique in cliques_data[k]]
	P_g = getPolyhedronFromCliques(cliques,degree)
	if not P_g.is_compact():
		continue
	normal_fan = P_g.normal_fan()
	for cone in normal_fan:
		M = np.array(cone.rays())
		if not TE.is_TE(M):
			result.append(graph_with_cliques(cliques))
			break
print(sage.graphs.graph_list.to_graph6(result))


