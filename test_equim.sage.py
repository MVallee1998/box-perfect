

# This file was *autogenerated* from the file ./test_equim.sage
from sage.all_cmdline import *   # import sage library

_sage_const_1 = Integer(1); _sage_const_0 = Integer(0)
import numpy as np
import TE
import tqdm

def getPolyhedronFromGraph(g):
	trivialConstraintsGe0 = np.array([np.zeros(g.order() +_sage_const_1 ) for i in range(_sage_const_0 ,g.order())])
	c = _sage_const_1 
	for j in trivialConstraintsGe0:
		j[c]=_sage_const_1 
		c = c+_sage_const_1 
	cliqueOfg = np.array(list(g.all_cliques()))
	cliqueConstraints = np.zeros((cliqueOfg.shape[_sage_const_0 ],g.order()+_sage_const_1 ))
	c=_sage_const_0 
	for K in cliqueOfg:
		cliqueConstraints[c][_sage_const_0 ] = _sage_const_1 
		cliqueConstraints[c][K+_sage_const_1 ] = -_sage_const_1 
		c=c+_sage_const_1 
	constraintvectors = np.concatenate((trivialConstraintsGe0, cliqueConstraints))
	return Polyhedron(ieqs = constraintvectors,base_ring=QQ)
	
f = open("perfect8.g6","r")

G = graphs_list.from_whatever(f)

N = len(G)
for k in tqdm.tqdm(range(N)):
	g = G[k]
	P_g = getPolyhedronFromGraph(g)
	if not P_g.is_compact():
		continue
	normal_fan = P_g.normal_fan()
	for cone in normal_fan:
		M = np.array(cone.rays())
		if not TE.is_TE(M):
			print(list(g.all_cliques()))
			break
	

