import numpy as np
from numba import jit
from itertools import combinations

#@jit(cache=True)
def det(M):
    M = [row[:] for row in M] # make a copy to keep original M unmodified
    N, sign, prev = len(M), 1., 1.
    for i in range(N-1):
        if M[i][i] == 0: # swap with another row having nonzero i's elem
            swapto = next( (j for j in range(i+1,N) if M[j][i] != 0.), None )
            if swapto is None:
                return 0 # all M[*][i] are zero => zero determinant
            M[i], M[swapto], sign = M[swapto], M[i], -sign
        for j in range(i+1,N):
            for k in range(i+1,N):
                M[j][k] = ( M[j][k] * M[i][i] - M[j][i] * M[i][k] ) * prev
        prev = M[i][i]
    res = sign * M[-1][-1]
    del M,sign,prev,N
    return res
    
def is_equimodular(M):
	k,n = M.shape
	first_not_null = 0.
	d=0.
	for iter_columns in combinations(range(n),k):
		d = det(M[:,iter_columns])
		if d!=0.:
			if not first_not_null:
				first_not_null = abs(d)
			if abs(d) != first_not_null:
				del d
				return False
	del d
	return(True)

def is_TE(M):
	k,n = M.shape
	for i in range(n,0,-1):
		for iter_rows in combinations(range(k),i):
			if not is_equimodular(M[iter_rows,:]):
				return False
	return True
