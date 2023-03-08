import numpy as np
import json
import TE
import tqdm
import multiprocessing
import networkx
degree = 8

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
	trivialConstraintsGe0 = np.array([np.zeros(n +1) for i in range(0,n)],dtype=np.float64)
	c = 1
	for j in trivialConstraintsGe0:
		j[c]=1.
		c = c+1
	cliqueConstraints = np.zeros((len(cliqueOfg),n+1),dtype=np.float64)
	c=0
	for K in cliqueOfg:
		cliqueConstraints[c][0] = 1.
		cliqueConstraints[c][K+1] = -1.
		c=c+1
	constraintvectors = np.concatenate((trivialConstraintsGe0, cliqueConstraints))
	P = Polyhedron(ieqs = constraintvectors,base_ring=QQ)
	del constraintvectors,trivialConstraintsGe0,cliqueConstraints
	return P
	
def read_file(filename):
    with open(filename, 'rb') as f:
        data = f.readlines()
    data = [x.strip() for x in data]
    return data



clique_path = "data_cliques/all_cliques_%d.txt" % degree
g6_path = "data_g6/perfect%d.g6" % degree


g6_data = [g6_byte.decode("utf-8") for g6_byte in read_file(g6_path)]
cliques_data = [json.loads(clique_bytes) for clique_bytes in read_file(clique_path)]
N = len(cliques_data)
list_cliques = [[np.array(clique) for clique in cliques_data[k]] for k in range(N)]

def is_bp(cliques):
	P_g = getPolyhedronFromCliques(cliques,degree)
	if not P_g.is_compact():
		del P_g
		return True
	normal_fan = P_g.normal_fan()
	for k in range(2,degree-1):
		for cone in normal_fan.cones(k):
			M = np.array(cone.rays(),dtype=np.float64)
			if not TE.is_equimodular(M):
				del P_g,normal_fan
				print('c')
				return False
			del M,cone
	print('c')
	del P_g,normal_fan
	return True

if __name__ == '__main__':
    pool = multiprocessing.Pool()
    pool = multiprocessing.Pool(processes=10)
    output = pool.map(is_bp, list_cliques)
    result = [graph_with_cliques(list_cliques[k]) for k in range(N) if not output[k]]
    print(sage.graphs.graph_list.to_graph6(result))


