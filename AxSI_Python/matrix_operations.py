# This file provide methods for matrix calculations

import numpy as np


def vector_len(v) -> int:
    ''' v could be either number or nd-array '''
    ''' if v is instance of ndarray, return its first dimension '''
    ''' else v has no shape attribute, treat it as an array of length 1 '''
    try:
        # if v is instance of nd-array
        length = v.shape[0]
    except:
        # v is a number
        length = 1
    return length


def normalize(v: np.ndarray) -> np.ndarray:
    ''' v is instance of nd-array '''
    ''' returns v divided by its euclidean norm'''
    if np.linalg.norm(v) > 0:
        return v / np.linalg.norm(v)
    return v


def atamult(A: np.ndarray, Y: np.ndarray, flag: int) -> np.ndarray:
    ''' multiply matrices A,Y according to a given flag'''
    if flag == 0:
        V = np.dot(A.T, np.dot(A, Y))   # V = Trans(A) @ A @ Y
    elif flag > 0:
        V = np.dot(A,Y)                 # V = A @ Y
    else:
        V = np.dot(A.T, Y)              # V = Trans(A) @ Y
    return V


def compute_w(H: np.ndarray, DM: np.ndarray, Z: np.ndarray) -> np.ndarray:
    ''' A series of matrix multiplication'''
    ''' Output: DM @ Trans(H) @ H @ DM @ Z '''
    w = np.dot(DM, Z)       # w  = DM @ Z
    ww = atamult(H, w, 0)   # ww = Trans(H) @ H @ DM @ Z
    w = np.dot(DM, ww)      # w  = DM @ Trans(H) @ H @ DM @ Z
    return w


def array_to_sym(arr: np.ndarray) -> np.ndarray:
    ''' Input: 1D array '''
    ''' expect len(arr) to be number of element in 
        upper triangle of a squared matrix '''
    ''' Output: 2D matrix of shape (size, size) '''
    ''' example for input: [1,2,3,4,5,6]:
        1, 2, 3
        2, 4, 5
        3, 5, 6
    '''
    size = get_mat_size_by_n_elements(len(arr))
    mat = np.zeros([size, size])
    mat[np.triu_indices(size)] = arr
    mat = mat.T
    mat[np.triu_indices(size)] = arr
    return mat


def get_mat_size_by_n_elements(n: int) -> int:
    ''' return the size of squared matrix
        with n elements in upper triangle '''
    ''' print message and return n if no such solution '''
    roots = np.roots([1, 1, -2 * n])
    pos_roots = roots[roots > 0]
    if pos_roots[0].is_integer():
        return int(pos_roots[0])
    else:
        print(f"could not convert array of len: {n} to symmetric matrix")
        return n

def sym_mat_to_array(mat: np.ndarray) -> np.ndarray:
    ''' Input: symmetric matrix'''
    ''' Output: 1D array with matrix elements '''
    ''' elements above diagonal are multiplied by 2 '''
    ''' for input 1, 2, 3  :
                  2, 4, 5
                  3, 5, 6
        output: [1, 2*2, 2*3, 4, 2*5, 6]
    '''
    mat = mat + np.triu(mat, k=1)         # multiply by two elements above diag
    arr = mat[np.triu_indices(3)]         # build array with elements of upper triangular
    return arr


def normalize_matrix_rows(mat: np.ndarray) -> np.ndarray:
    ''' input: 2D matrix '''
    ''' normalize each row in input separately'''
    ''' norm_mat.shape == mat.shape '''
    norm_mat = np.zeros(mat.shape)          # init new matrix of same shape
    rows_norm = np.linalg.norm(mat, axis=1) # compute array of norm values of each row
    # divide each row by its own norm, ignoring zero rows
    norm_mat[rows_norm > 0] = mat[rows_norm > 0] / rows_norm[rows_norm > 0][np.newaxis].T
    return norm_mat


def exp_vec_mat_multi(vec: np.ndarray, mat: np.ndarray, coeff: float, axis=0) -> np.ndarray:
    ''' A series of matrix and vector multiplications and exponents '''
    ''' Output: e^(coeff * vec @ mat @ trans(vec)) '''
    multi_mat_vec = np.dot(mat, vec.T)                              # mat @ trans(vec)
    multi_vec_mat_vec = np.sum(multi_mat_vec * vec.T, axis=axis)    # vec @ mat @ trans(vec)
    exp = np.exp(coeff * multi_vec_mat_vec)                         # e^(c * vec @ mat @ trans(vec))
    return exp
