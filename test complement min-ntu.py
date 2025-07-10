import numpy as np
from itertools import combinations,permutations, product
import tqdm

def is_min_nonTU(matrix):
    n, m = matrix.shape
    if not (np.all(np.sum(matrix,axis=0)%2==0) and np.all(np.sum(matrix,axis=1)%2==0) and np.sum(matrix)%4==2):
        return False
    if round(np.linalg.det(matrix)) not in [2,-2]:
        return False
    for size in range(1, n):
        for row_comb in combinations(range(n), size):
            for col_comb in combinations(range(n), size):
                submatrix = matrix[np.ix_(row_comb, col_comb)]
                det = round(np.linalg.det(submatrix))
                if det not in [-1, 0, 1]:
                    return False
    return True

def is_min_nonTU_camion(matrix):
    n, m = matrix.shape
    # if not (np.all(np.sum(matrix,axis=0)%2==0) and np.all(np.sum(matrix,axis=1)%2==0) and np.sum(matrix)%4==2):
    #     return False
    if round(np.linalg.det(matrix)) not in [2,-2]:
        return False
    return camion_test(matrix)
    for size in range(1, n):
        for row_comb in combinations(range(n), size):
            for col_comb in combinations(range(n), size):
                submatrix = matrix[np.ix_(row_comb, col_comb)]
                det = round(np.linalg.det(submatrix))
                if det not in [-1, 0, 1]:
                    return False
    return True

def is_TU(matrix):
    n, m = matrix.shape
    # if not (np.all(np.sum(matrix,axis=0)%2==0) and np.all(np.sum(matrix,axis=1)%2==0) and np.sum(matrix)%4==2):
    #     return False
    if round(np.linalg.det(matrix)) not in [1,0,-1]:
        # print("hello", np.linalg.det(matrix))
        return False
    for size in range(1, n):
        for row_comb in combinations(range(n), size):
            for col_comb in combinations(range(n), size):
                submatrix = matrix[np.ix_(row_comb, col_comb)]
                det = round(np.linalg.det(submatrix))
                if det not in [-1, 0, 1]:
                    # print(det)
                    return False
    return True

def col_complement(M,i):
    n=M.shape[0]
    N = M.copy()
    for k in range(n):
        if k!=i:
            N[:,k] = (N[:,k] + N[:,i]) % 2
    return(N)
def row_complement(M,i):
    n=M.shape[0]
    N=M.copy()
    for k in range(n):
        if k!=i:
            N[k,:] = (N[k,:] + N[i,:]) % 2
    return(N)

def construct_odd_cycle(n):
    M = np.eye(n)
    for k in range(n):
        M[k,(k+1)%n] = 1
    return M

# n= 6

# M = construct_odd_cycle(n)

# M = np.array([[1,1,0,0,0,0,0],
#              [0,1,1,1,1,1,1],
#              [0,0,1,0,1,1,1],
#              [0,0,0,1,1,1,1],
#              [0,1,0,1,1,0,1],
#              [0,0,0,0,0,1,1],
#              [1,1,0,1,0,0,1]])

# M = np.array([[1,1,1,1,1,1],
#               [0,1,0,1,1,1],
#               [0,0,1,1,1,1],
#               [1,0,1,1,0,1],
#               [0,0,0,0,1,1],
#               [1,0,1,0,0,1]])

# N = (-1)*np.ones((7,7))
# N[0,:] = 1
# N[1:,1:] += 2*M
# N*=1/2
# print(N)
# def construct_basis_2(n):
#     list_basis_2=[]
    



def is_complement_mintu(M):
    if not is_min_nonTU(M):
        return False
    is_complement_mintu = True
    n=M.shape[0]
    for i in range(n):
        if not is_min_nonTU(col_complement(M,i)):
            return False
        if not is_min_nonTU(row_complement(M,i)):
            return False
        

    for i in range(n):
        for j in range(n):
            N=row_complement(col_complement(M,i),j)
            if not is_min_nonTU(N):
                return False
            
    return True

def is_complement_mintu_camion(M):
    if not is_min_nonTU_camion(M):
        return False
    is_complement_mintu = True
    n=M.shape[0]
    for i in range(n):
        if not is_min_nonTU_camion(col_complement(M,i)):
            return False
        if not is_min_nonTU_camion(row_complement(M,i)):
            return False
        

    for i in range(n):
        for j in range(n):
            N=row_complement(col_complement(M,i),j)
            if not is_min_nonTU_camion(N):
                return False
            
    return True

# for A in [np.array([[1,0,0,0,0,0],
#                     [1,1,1,0,0,0]]),
#           np.array([[1,0,0,0,0,0],
#                     [1,1,1,1,1,0]]),
#           np.array([[1,1,1,0,0,0],
#                     [1,1,1,1,1,0]])]:
#     for x in range(2**(24)):
#         B = np.reshape(np.array([int(i) for i in bin(x)[2:].zfill(24)],dtype=int), (4,6))
#         M = np.zeros((7,7))
#         M[0,1:]=1
#         M[[1,2],0] = 1
#         M[1:3,:][:,1:] = A
#         M[3:,:][:,1:] = B
#         if is_complement_mintu(M):
#             print('Hello')

# def build_signed_perm(n,perm,x):
#     S = np.zeros((n,n))
#     for i in range(n):
#         S[i,perm[i]]= (-1)**((x >> i) & 1)
#     return S
# M = np.array([[1,1,0,0,0,0,0],
#              [0,1,1,1,1,1,1],
#              [0,0,1,0,1,1,1],
#              [0,0,0,1,1,1,1],
#              [0,1,0,1,1,0,1],
#              [0,0,0,0,0,1,1],
#              [1,1,0,1,0,0,1]])
# M_inv = np.linalg.inv(M)
# C7 = construct_odd_cycle(7)
# total = 0
# counter = 0
# for k in tqdm.tqdm(range(2**7)):
#     for perm in permutations(range(7)):
#         counter+=1
#         S = build_signed_perm(7,perm,k)
#         # print(S)
#         C=np.dot(C7,S)
#         if is_TU(np.dot(C,N)):
#             total +=1

# print("Percentage:", round((total/counter)*10000)/100,'%') #1.16%

def is_eulerian(matrix):
    """Check if a square 0/1 matrix has even row and column sums."""
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    return np.all(row_sums % 2 == 0) and np.all(col_sums % 2 == 0)

def camion_test(matrix):
    """
    Camion's criterion for TU of 0/1 matrices:
    Every Eulerian square submatrix has determinant divisible by 4.
    """
    rows, cols = matrix.shape
    min_dim = min(rows, cols)
    for k in range(2, min_dim):  # size of square submatrix
        for row_idx in combinations(range(rows), k):
            for col_idx in combinations(range(cols), k):
                sub = matrix[np.ix_(row_idx, col_idx)]
                if is_eulerian(sub):
                    det = round(np.linalg.det(sub))
                    if det % 4 != 0:
                        return False
    return True

def generate_all_binary_matrices(n):
    """
    Generate all n x n binary (0/1) matrices.
    """
    for flat in product((0, 1), repeat=(n-1)*(n-1)):
        A=np.array(flat).reshape((n-1, n-1))
        B =np.zeros((n,n))
        B[:n-1,:n-1] = A
        B[n-1,:n-1] = np.sum(A,axis=0)
        B[:n-1,n-1] = np.sum(A,axis=1)
        B[n-1,n-1] = np.sum(B[n-1,:n-1])%2
        if np.any(np.sum(B,axis=0)%2!=0) or np.any(np.sum(B,axis=1)%2!=0) or np.sum(B)%4!=2:
            continue
        yield B

def generate_all_bin(n):
    database=[]
    for flat in product((0,1),repeat=n):
        flat_np=np.array(flat)
        if np.sum(flat_np)%2==0 and np.any(flat_np==1):
            database.append(flat)
    for A_combi in tqdm.tqdm(combinations(database[1:],n-2)):
        A = np.array(A_combi)
        A = np.vstack([A,np.sum(A,axis=0)%2,database[0]])
        if np.sum(A)%4==2:
            yield A

for M in generate_all_bin(9):
    if is_complement_mintu_camion(M):
        print(M,np.linalg.det(M))
    

