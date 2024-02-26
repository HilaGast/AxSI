# This file provides methods and basic functionality for AxSI analysis

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import gamma
from multiprocessing import managers


# parameters for later use
NUM_OF_PROCESSES = 2                # use parallel iff NUM_OF_PROCESSES > 1
ADD_VALS = np.arange(0.1, 32, 0.2)  # [0.1, 0.3, 0.5, ... , 31.9]
N_SPEC = len(ADD_VALS)              # N_SPEC == 160


class Values:
    ''' A basic class to be extended with different attributes '''
    ''' Provide get/set methods base on given attribute '''
    ''' Used for creating a shared object in parallel prcoessing '''
    
    def get(self, attr, i=-1):
        ''' assume self uses "attr" to hold array '''
        ''' return element if i >= 0, otherwise the whole array'''
        if i==-1:
            # return array
            return getattr(self, attr)
        # return specific element by index
        return getattr(self, attr)[i]
    
    def set(self, attr, new_val, i=-1):
        ''' assume self uses "attr" to hold array '''
        ''' set element if i >= 0, otherwise replace the whole array'''
        if i==-1:
            # replace array
            setattr(self, attr, new_val)
        else:
            # set specific element by index
            getattr(self, attr)[i] = new_val

    @staticmethod
    def copy_values(source, target, attributes):
        ''' assume source and target holds the same attribute'''
        for attr in attributes:
            # replace array in target with source array
            target.set(attr, source.get(attr))

    def get_dict(self, attributes):
        ''' assume self holds data for each attr in attributes '''
        ''' return dictionary with attributes as keys and data as values'''
        d = {attr: self.get(attr) for attr in attributes}
        return d
    
    def nonzero(self, attr):
        ''' for given attribute in self:
        set 0 instead of every negative value '''
        arr = self.get(attr)
        arr[arr <= 0] = 0
        self.set(attr, arr)


# init a new manager for parallel processing 
class MyManager(managers.BaseManager):
    pass


def cart2sph_env(v: np.ndarray) -> tuple:
    ''' An envelope function for specific use'''
    ''' Expect len(v) == 3 '''
    ''' Change the sign of the third element '''
    return cart2sph(v[0], v[1], -v[2])


def cart2sph(x: float, y: float, z: float) -> tuple:
    ''' convert from 3D cartesian representation to spherical '''
    xy = np.sqrt(x ** 2 + y ** 2)  # sqrt(x² + y²)
    r = np.sqrt(xy ** 2 + z ** 2)  # r = sqrt(x² + y² + z²)
    phi = np.arctan2(z, xy)
    theta = np.arctan2(y, x)

    return r, theta, phi


def smooth4dim(data: np.ndarray) -> np.ndarray:
    ''' Input: 4D array '''
    ''' Output 4D array of the same size as input '''
    ''' smooth each 3D slice based on last dimension '''
    new_data = np.zeros(data.shape)
    for i in range(data.shape[3]):
        new_data[:,:,:,i] = smooth_data_slice(data[:,:,:,i])
    return new_data


def smooth_data_slice(slice: np.ndarray) -> np.ndarray:
    ''' use gaussian filter to smooth input data with pre-defined parameters '''
    smoothed_slice = gaussian_filter(slice, sigma=0.65, truncate=3, mode='nearest')
    return smoothed_slice


def init_yd():
    ''' Initialization of array for calc_axsi '''
    ''' yd.shape == ADD_VALS.shape '''
    alpha = 3
    beta = 2
    gamma_pdf = gamma.pdf(ADD_VALS, a=alpha, scale=beta)
    yd = gamma_pdf * np.pi * (ADD_VALS / 2)**2
    yd = yd / np.sum(yd)
    return yd


def init_l_matrix(n):
    ''' Initialization of array for calc_axsi '''
    ''' l_mat.shape == (n, n+2)'''
    ''' for n == 3 output is: 
         1, 0, 0, 0, 0
        -1, 1, 0, 0, 0
         0,-1, 1, 0, 0
    '''
    ones = np.ones(n)
    # create ID matrix with a diagonal of -1 below the main diag
    l_mat = np.eye(n) - np.tril(ones, -1) * np.triu(ones, -1)
    # add two columns of 0
    l_mat = np.append(l_mat, np.zeros((n, 2)), axis=1)
    return l_mat