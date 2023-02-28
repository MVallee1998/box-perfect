import numpy as np
import TE
import tqdm

def getPolyhedronFromGraph(g):
	trivialConstraintsGe0 = np.array([np.zeros(g.order() +1) for i in range(0,g.order())])
	c = 1
	for j in trivialConstraintsGe0:
		j[c]=1
		c = c+1
	cliqueOfg = np.array(list(sage.graphs.cliquer.all_cliques(g)))
	cliqueConstraints = np.zeros((cliqueOfg.shape[0],g.order()+1))
	c=0
	for K in cliqueOfg:
		cliqueConstraints[c][0] = 1
		cliqueConstraints[c][K+1] = -1
		c=c+1
	constraintvectors = np.concatenate((trivialConstraintsGe0, cliqueConstraints))
	return Polyhedron(ieqs = constraintvectors,base_ring=QQ)
	
f = open("perfect7.g6","r")

G = graphs_list.from_whatever(f)

N = len(G)
for k in tqdm.tqdm(range(N)):
	g = G[k]
	print(g.adjacency_matrix())
	P_g = getPolyhedronFromGraph(g)
	if not P_g.is_compact():
		continue
	normal_fan = P_g.normal_fan()
	for cone in normal_fan:
		M = np.array(cone.rays())
		if not TE.is_TE(M):
			print(list(g.all_cliques()))
			break

