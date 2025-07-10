import TE
import numpy as np
from itertools import combinations
# B = np.array([[-1,1,1,1,1,1,1,1],
#               [1,1,1,-1,-1,-1,-1,-1],
#               [1,-1,1,1,1,1,1,1],
#               [1,-1,-1,1,-1,1,1,1],
#               [1,-1,-1,-1,1,1,1,1],
#               [1,-1, 1,-1, 1, 1,-1, 1],
#               [1,-1,-1,-1,-1,-1, 1, 1],
#               [1, 1, 1,-1, 1,-1,-1, 1]
#               ])
# C=4*np.linalg.inv(B)
# C[:,5] *=-1
# C[:,6] *=-1
# C[3,:] *=-1
# C[4,:] *=-1
# print(C)
# C = ((np.ones((8,8))+B)/2)[1:,1:]
# print(np.linalg.eig(B))
#print(4*np.linalg.inv(B))
#print(np.linalg.det(M))
#print(TE.is_TE(B),TE.is_TE(B.T))

# D = np.array([[1,1,1,1,1,1],
#               [1,1,1,1,-1,-1],
#               [1,1,1,-1,-1,1],
#               [1,1,-1,-1,1,1],
#               [1,-1,-1,1,1,1],
#               [1,-1,1,1,1,-1]])

# print(TE.is_TE(D))

# N = np.array([[1,1,1,1],
#               [1,0,1,1],
#               [1,1,1,0],
#               [1,1,0,1],])

# print(np.linalg.det(N))

# M = np.array([[0,1,1,1,1],
#               [1,1,1,1,1],
#               [1,1,1,1,0],
#               [1,1,1,0,1],
#               [1,1,0,1,1]])



# for i in range(6):
#      for comb1 in combinations(range(5),i):
#           arr1 = np.array(comb1)
#           for j in range(5):
#                for comb2 in combinations(range(1,5),j):
#                     arr2=np.array(comb2)
#                     M[0,:] = 0
#                     M[0,comb1]= 1
#                     M[1:,0] = 0
#                     M[comb2,0]=1
#                     L=[]
#                     if np.abs(np.linalg.det(M))>1:
#                         for I_comb in combinations(range(5),4):
#                             I=np.array(I_comb)
#                             for J_comb in combinations(range(5),4):
#                                 J=np.array(J_comb)
#                                 D = np.abs(np.linalg.det(M[I,:][:,J]))
#                                 if D!=0 and D not in L:
#                                     L.append(D)
#                         if len(L)==1:
#                             print(M,L,np.linalg.det(M))

# N = np.array([[0,0,0,0,0,0],
#               [0,0, 0, 0 ,1, 0],
#  [0,0, 1, 1, 1, 1],
#  [0,0, 1, 1, 1, 0],
#  [0,1, 1, 1, 0, 1],
#  [0,0, 1, 0, 1, 1]])

# for i in range(7):
#      for comb1 in combinations(range(6),i):
#           arr1 = np.array(comb1)
#           for j in range(6):
#                for comb2 in combinations(range(1,6),j):
#                     arr2=np.array(comb2)
#                     N[0,:] = 0
#                     N[0,comb1]= 1
#                     N[1:,0] = 0
#                     N[comb2,0]=1
#                     L=[]
#                     if np.abs(np.linalg.det(N))> 1:
#                         for I_comb in combinations(range(6),5):
#                             I=np.array(I_comb)
#                             for J_comb in combinations(range(6),5):
#                                 J=np.array(J_comb)
#                                 D = np.abs(np.linalg.det(N[I,:][:,J]))
#                                 if D!=0 and D not in L:
#                                     L.append(D)
#                         if len(L)==1 and np.abs(np.linalg.det(N))!=1:
#                             print(N,L,np.linalg.det(N))


def search(M,max_depth):
    if np.linalg.det(M)==0:
        print("singular matrix")
    n= M.shape[0]
    if n<max_depth:        
        N = np.zeros((n+1,n+1))
        N[1:,1:] = M
        for i in range(n+2):
            for comb1 in combinations(range(n+1),i):
                for j in range(n+1):
                    for comb2 in combinations(range(1,n+1),j):
                            N[0,:] = 0
                            N[0,comb1]= 1
                            N[1:,0] = 0
                            N[comb2,0]=1
                            L=[]
                            if np.abs(np.linalg.det(N))> 0.00001:
                                for I_comb in combinations(range(n+1),n):
                                    I=np.array(I_comb)
                                    for J_comb in combinations(range(n+1),n):
                                        J=np.array(J_comb)
                                        D = np.abs(np.linalg.det(N[I,:][:,J]))
                                        if D!=0 and D not in L:
                                            L.append(D)
                                if len(L)==1:
                                    if np.abs(np.linalg.det(M))>1:
                                        print(N,L,np.linalg.det(N))
                                    search(N,max_depth)
                                        

N = np.array([[1,1,1,1,1,1],
              [1,0,0,0,1,1],
              [1,0,0,1,1,0],
              [1,0,1,1,0,0],
              [1,1,1,0,0,0],
              [1,1,0,0,0,1],])
M = 2*N + np.ones((6,6))


# N = np.array([[1,0,1,0,1],
#               [1,1,1,1,1],
#               [1,1,0,0,1],
#               [1,1,0,1,0],
#               [1,1,1,0,0]])
# search(N,9)
# print(np.linalg.det(M),L)