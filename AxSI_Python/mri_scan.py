# This file defines MRI scan object for AxSI analysis
# Scan obect holds various objects for Data, bvals, bvecs and mask

import os
import numpy as np
import nibabel as nb
from pathlib import Path
from dipy.io import read_bvals_bvecs
from toolbox import cart2sph_env, smooth4dim
from matrix_operations import normalize_matrix_rows


class Scan_File:
    ''' A basic class holds a file path and its data loaded'''
    def __init__(self, filepath: Path, raw_data: np.ndarray) -> None:
        self.filepath = filepath
        self.raw_data = raw_data    # data is already loaded


class NIFTIFile(Scan_File):
    ''' A class holds nifti file such as scan data and mask'''
    def __init__(self, filepath: Path) -> None:
        self.diff_file = nb.load(filepath)
        # load data for parent class
        super().__init__(filepath, self.diff_file.get_fdata())


class Scan:
    ''' Main class for scan object use in the analysis '''
    ''' holds relevant data from files and parameters '''
    ''' provides get functions to retreive data '''
    def __init__(self, file_names: dict, *params) -> None:
        ''' filenames should contain path to four files by specific keys'''
        ''' params should be ordered in specific way '''
        # load data of bvals and bvecs files
        bvals, bvecs = read_bvals_bvecs(str(file_names['bval']), str(file_names['bvec']))
        self.bval = BVAL(file_names['bval'], bvals)             # init BVAL object
        self.bvec = BVEC(file_names['bvec'], bvecs, self.bval)  # init BVEC object
        self.data = DATA(file_names['data'], self.bval)         # init DATA object
        self.mask = MASK(file_names['mask'])                    # init MASK object
        self.shape = self.mask.raw_data.shape                   # based on scan resolution
        self.num_of_vols = np.prod(self.shape)                  # number of voxels in scan
        self.param_dict = self.build_param_dict(*params)


    def build_param_dict(self, small_delta: int, big_delta: int, gmax: float, gamma_val: float) -> dict:
       ''' create a dictionary with scan parameters '''
       grad_dirs = self.bvec.grad_dirs
       # init dictionary
       scan_param = {'nb0': self.bval.locs[0], 'small_delta': small_delta, 'big_delta': big_delta, 'gmax': gmax}
       max_q = gamma_val * small_delta * gmax / 10e6
       # convert grad_dirs to spherical representation
       r_q, phi_q, theta_q = cart2sph_env(grad_dirs.T)
       r_q = r_q * max_q                            # multiply by coefficient
       scan_param['q_dirs'] = grad_dirs * max_q     # multiply by coefficient
       # set spherical representations
       scan_param['r'] = r_q
       scan_param['phi'] = phi_q
       scan_param['theta'] = theta_q
       return scan_param
    
    
    def get_smoothed_data(self, row_index=None, col_index=None) -> np.ndarray:
        ''' method to get smoothed signal data of mri scan '''
        ''' data is flattened: dim1 are the voxels, dim2 time '''
        ''' pre-filter for brain voxels '''
        data = self.data.smoothed[self.mask.index]
        if row_index:         # for taking data of specific voxels
            data = data[row_index, :]
        if col_index:         # for taking data of specific bvalue
            data = data[:, col_index]
        return data
    

    def get_signal_0(self, take_first: bool=False) -> np.ndarray:
        ''' Input: flag for DTI '''
        ''' for DTI1000 use only the first appearance of bvalue==0
            for shellDTI use the mean of all bvalue==0 '''
        ''' Output: 1D array of signal when bvalue==0 '''
        if take_first:
            # first image when bvalue==0
            return self.data.signal_0[self.mask.index]
        else:
            # mean over bvalue==0
            return self.data.signal_0_mean[self.mask.index]
        
    def get_locs_by_bvalues(self, bvalue) -> list:
        ''' get a list of indices to images taken with bvalue '''
        return self.bval.locs[bvalue]
    
    def get_shape(self) -> np.ndarray:
        ''' get brain resolution '''
        return self.shape
    
    def get_num_of_voxels(self) -> int:
        ''' return value is number of brain voxels
            not as the attribute num_of_vol kept in scan ''' 
        return len(self.mask.index)
        
    def get_params(self) -> dict:
        ''' get dictionary of scan parameters '''
        return self.param_dict

    def get_bval_data(self) -> np.ndarray:
        ''' get array of bval real values '''
        return self.bval.data * 1000
    
    def get_bval_length(self) -> int:
        ''' get number of timeframes '''
        return self.bval.length
    
    def get_shell(self) -> np.ndarray:
        ''' Output: array with unique nonzero elements of bvalues '''
        return self.bval.shell
    
    def get_max_bval(self) -> float:
        ''' Output: maximum value of bval '''
        return self.bval.max_bval
    
    def get_grad_dirs(self) -> np.ndarray:
        ''' Output: array with bvec grad_dirs data '''
        return self.bvec.grad_dirs
    
    def get_bvec_norm(self) -> np.ndarray:
        ''' Output: 2D array with normalized bvec data '''
        return self.bvec.norm_data
    

    def reshape_to_brain_size(self, data: np.ndarray) -> np.ndarray:
        ''' assuming data.shape[0] == len(mask.index) '''
        ''' data can be n-dim '''
        exshape = data.shape[1:]    # shape without first dimension, could be empty tuple
        new_data = np.zeros((self.num_of_vols, ) + exshape) # concat tuples
        new_data[self.mask.index] = data
        new_data = new_data.reshape(self.shape + exshape)   # concat tuples
        return new_data


    def save_files(self, files: dict[str, np.ndarray], subj_folder: Path) -> None:
        ''' save nifti files in subj_folder based on files dictionary'''
        ''' dict should hold strings as keys and np.ndarray as values '''
        for key in files.keys():
            filename = build_file_name(subj_folder, key)    # build full path to file
            data = self.reshape_to_brain_size(files[key])   # plant brain data in array of original image dimension
            save_nifti(filename, data, self.data.diff_file.affine)


def save_nifti(fname, img, affine) -> None:
    ''' save img as nifti file '''
    ''' fname determine the path to file '''
    file_img = nb.Nifti1Image(img, affine)
    nb.save(file_img, fname)


def build_file_name(subj_folder, file_name) -> str:
    ''' build path to save files in a specific manner '''
    return f'{subj_folder}{os.sep}AxSI{os.sep}{file_name}.nii.gz'



class BVAL(Scan_File):
    ''' Object to store data of bvals file '''
    def __init__(self, filepath: Path, raw_data: np.ndarray) -> None:
        # call parent constructor with path and raw_data 
        super().__init__(filepath, raw_data)
        # map values to rounded integers
        data = self.compute_rounded_data()
        # find indices of not rounded integers
        self.low_locs = self.find_low_locs(data)
        # remove instances of not rounded integers
        data = np.delete(data, self.low_locs)
        # get unique values of bvalue
        shell = np.unique(data)
        self.length = len(data)
        self.data = data
        self.shell = shell[shell > 0]               # keep only nonzero bvalues
        self.locs = {value: [] for value in shell}  # init dictionary of shell and indices
        self.fill_locations_dict()
        self.max_bval = np.max(data)                # maximal bvalue 
        self.norm = np.sqrt(data / self.max_bval)   # normalize bvalue data


    def compute_rounded_data(self) -> np.ndarray:
        ''' round bvalues to find bad values '''
        ''' for example: 1995 => 2, 1500 => 1.5 '''
        bval2 = 2 * np.asarray(self.raw_data)   # multiply by 2
        rounded_data = bval2.round(-2) / 2000   # round last two digits, divide by 2000
        return rounded_data


    def fill_locations_dict(self) -> None:
        ''' store for each bvalue the indices of relevant images '''
        data = self.data
        for i in range(self.length):
            # data[i] is a bvalue, use it as key
            self.locs[data[i]].append(i)


    def find_low_locs(self, bval) -> None:
        ''' get locations where 0 < elements < 1 '''
        low_locs = np.intersect1d(np.where(bval > 0)[0], np.where(bval < 1)[0])
        return low_locs


class BVEC(Scan_File):
    ''' Object to store data of bvecs file '''
    def __init__(self, filepath: Path, raw_data: np.ndarray, bval: BVAL) -> None:
        # call parent constructor with path and loaded data
        super().__init__(filepath, raw_data)
        # use bval to reshape data if needed
        data = np.reshape(self.raw_data, [bval.length, -1])
        # use bval to omit bad values
        data = np.delete(data, bval.low_locs, 0)
        # keep data multiplited by bvals coefficient
        self.grad_dirs = data * bval.norm[np.newaxis].T
        # normalize data
        self.norm_data = normalize_matrix_rows(data)    
        self.norm_grad_dirs = normalize_matrix_rows(self.grad_dirs)



class DATA(NIFTIFile):
    ''' Object to store signal data of mri scan '''
    ''' assuming preprocessed for now '''
    def __init__(self, filepath: Path, bval: BVAL) -> None:
        # call parent constructor with path
        super().__init__(filepath)
        data = np.asarray(self.raw_data, dtype='float64')
        # use bval to omit bad values
        data = np.delete(data, bval.low_locs, 3)
        data[data <= 0] = 0.0
        X, Y, Z, n = data.shape     # assume data has 4 dimensions
        self.corrected = data
        smoothed_data = smooth4dim(data)    # smooth every image separately
        self.smoothed = smoothed_data.reshape((X*Y*Z, n))   # flatten the brain to make data 2D
        self.signal_0 = self.smoothed[:, bval.locs[0][0]]   # keep the signal when bvalue=0 for the first time
        self.signal_0_mean = np.nanmean(self.smoothed[:, bval.locs[0]], axis=1) # mean across bvalue=0
        

class MASK(NIFTIFile):
    ''' Object to store brain mask file '''
    def __init__(self, filepath) -> None:
        # call parent constructor with path
        super().__init__(filepath)
        self.flattened = self.raw_data.flatten()        # flatten the images to be 1D
        self.index = np.where(self.flattened > 0)[0]    # indices for voxels of brain