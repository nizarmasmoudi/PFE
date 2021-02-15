import numpy as np

def vsplit(mat, n_splits):
    mat = np.array(mat)
    l = mat.shape[0]
    if l%n_splits > 0:
        arr = np.vsplit(mat[:-(l%n_splits), ...], n_splits)
        arr[-1] = np.append(arr[-1], mat[-(l%n_splits):, ...], axis=0)
        return arr
    else:
        return np.vsplit(mat, n_splits)