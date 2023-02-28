import numpy as np
from itertools import combinations
import numba

def getEssencePolyhedronFromGraph(g):
	trivialConstraintsGe0 = np.array([np.zeros(g.order()*g.order() +1) for i in range(0,g.order()*g.order())])
	trivialConstraintsLe1 = np.array([np.zeros(g.order()*g.order() +1) for i in range(0,g.order()*g.order())])
	c = 1
	for j in trivialConstraintsGe0:
		j[c]=1
		c = c+1
	
	c = 1
	for j in trivialConstraintsLe1:
		j[c]=-1
		j[0] = 1
		c = c+1
	
	cliqueOfg = list(g.all_cliques())
	
	cliqueConstraints = np.array([np.zeros(g.order()* g.order()+1) for j in cliqueOfg])
	c=0
	for K in cliqueOfg:
		cliqueConstraints[c][0] = 1
		for v in K:
			for i in range(0,g.order()):
				cliqueConstraints[c][i*g.order() + v+1] = -1
		c=c+1
	constraintvectors = np.concatenate((trivialConstraintsGe0, cliqueConstraints,trivialConstraintsLe1))
	print("coucou")
	return Polyhedron(ieqs = constraintvectors)
			


f = open("perfect5.g6","r")

G = graphs_list.from_whatever(f)

for g in G:
	if(not getEssencePolyhedronFromGraph(g).is_lattice_polytope()):
		P = g.plot()
		P.show()




