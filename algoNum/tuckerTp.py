#!/usr/bin/python3

import h5py # to open hdf5 data
# import pillow # image processing
import numpy as np
import scipy # (gaussian_filter)
import tensorly as tl # to manipulate tensors (representations and operations)
from tensorly import tenalg as tla

from tucker_tools import build_tensor

# `pip install -U tensorly`
# `conda install -c tensorly tensorly`

def loadTensor1():
    filename = "data/CPL-40Ma-2X-ICE_SE_3805_3854_T2M_PRECIP.nc"
    data = {}
    h5 = h5py.File(filename)
    for d in h5:
        data[d] = h5[d]
    return data["t2m"][:] - 273.15

"""
    X = un Tenseur
    r = multilinéaire
"""
def truncatedHOSVD(X, r):
    Us = []
    d = r.size
    for k in range(d):
        # la k-matricisation du tenseur X
        Xᵏ = tl.unfold(X, k)
        
        # SVD tronquée à l'ordre rₖ
        U, Σ, V = np.linalg.svd(Xᵏ, full_matrices=False)
        print(U)
        print(Σ)
        print(V)
        # Us[k] =  
        
        

def main():
    # n = ??? # tensors mode (numpy array of size d)
    
    # X = loadTensor("data/minst.h5")
    X = loadTensor1()
    print("Tensor shape:", X.shape)
    
    # multilinear rank (numpy array of size d)
    r = np.array((12, 40, 40))
    print("Tensor multilinear rank:", r)
    truncatedHOSVD(X, r)

if __name__ == "__main__" :
    main()