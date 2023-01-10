#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27

 Tucker tools
 
@author: O. Coulaud
"""
#
import numpy as np
#  tensor class
#   http://tensorly.org/stable/user_guide/
#from tensorly import tucker_to_tensor
import tensorly as tl
from tensorly import tenalg as tla
import h5py

def load_data():
        """
                Load data from .nc file
                Return
                        data:                           dict of numpy arrays of floats
                        attributes: dict of numpy arrays of bytes strings
        """
        filename        = "/home/coulaud/tensor/tp/CPL-40Ma-2X-ICE_SE_3805_3854_T2M_PRECIP.nc"
        data = {}
        h5=h5py.File(filename)
        for d in h5:
            data[d] = h5[d]
        #data["t2m"][:] - 273.15
        return data["t2m"][:] - 273.15 #data,h5.attrs
        
def build_tensor(n, r):
    '''
    Generate a tensor of dimension n with rank r
    Input
        n  the dimension of the mode of the tensor
            numpy array of size d 
        r  the multilinear rang of the tensor
            numpy array of size d 
    Outpout 
        the tensor numpy array of size n 
    '''

    d = n.size
    if (d != r.size):
        raise ValueError(" size of r != n")
    r_size = 1
    for i in r:
        r_size *= i
    core = 1/(10 + np.arange(1, r_size + 1)**3)
    core = core.reshape(r)
    # Generate the factors
    factors = []
    print("Build factors")
    for i in range(d):
        U = np.random.uniform(size=n[i]*r[i]).reshape([n[i], r[i]])
        U, R = np.linalg.qr(U, mode='reduced')
        factors.append(U)
    X = tl.tucker_to_tensor([core, factors])

    return X
