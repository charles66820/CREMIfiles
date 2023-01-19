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
    d = r.size
    U_ = [None] * d
    for k in range(d):
        # la k-matricisation du tenseur X
        Xᵏ = tl.unfold(X, k)

        # SVD tronquée à l'ordre rₖ
        # U, Σ, V = np.linalg.svd(Xᵏ, full_matrices=False)
        # U_[k] = U[:,0:r[k]]
        U, Σ, V = tl.partial_svd(Xᵏ)
        U_[k] = U

    S = tla.multi_mode_dot(X, U_, transpose=True)

    return S, U_

def checkAndPrintDiff(X, Xref):
    norm = tl.norm(Xref-X)/tl.norm(Xref)
    print("norm for X and Xref:", norm)

def main():
    # ordre du tenseur
    d = 3
    # dimension du tenseur. tensors mode (numpy array of size d ??)
    n = np.asarray([8,8,8])
    # rang du tenseur. multilinear rank (numpy array of size d)
    r= np.asarray([4,3,2])
    # r = np.array((12, 40, 40))
    # Construction d'un tenseur, X, aléatoire d'ordre d et de rang r
    X = build_tensor(n, r)

    # X = loadTensor("data/minst.h5")
    # X = loadTensor1()

    print("Tensor shape:", X.shape)
    print("Tensor multilinear rank:", r)

    S, U = truncatedHOSVD(X, r)
    Xref = tl.tucker_to_tensor((S, U))
    checkAndPrintDiff(X, Xref)


if __name__ == "__main__" :
    main()