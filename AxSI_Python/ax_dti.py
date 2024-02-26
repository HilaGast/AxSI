# This file is for running DTI for AxSI analysis

import numpy as np
from pathlib import Path
from multiprocessing import Pool
from toolbox import Values, MyManager, NUM_OF_PROCESSES
from mri_scan import Scan
from matrix_operations import array_to_sym, sym_mat_to_array


EPS = np.finfo('float').eps                     # 2.22X10-16
KEYS = ['dt', 'md', 'fa', 'eigval', 'eigvec']   # attributes of DTI calculations


class DTI_Values(Values):
    ''' An extended class for Values with 
        relevant attributes for DTI '''
    def __init__(self, n) -> None:
        self.fa = np.zeros(n)
        self.md = np.zeros(n)
        self.dt = np.zeros((n, 3, 3))
        self.eigval = np.zeros((n, 3))
        self.eigvec = np.zeros((n, 3))


# register class to MyManager, used for multiprocessing
MyManager.register('DTI_Values', DTI_Values)


class AXSI_DTI:
    ''' The main object for performing DTI '''
    def __init__(self, scan: Scan, bvalue: float, is1000: bool) -> None:
        ''' Use scan for initialization, don't keep it '''
        self.num_of_vols = scan.get_num_of_voxels()             # number of brain voxels in mask
        self.values = DTI_Values(self.num_of_vols)              # init struct with zero values
        # compute in advance
        bvlocs = scan.get_locs_by_bvalues(bvalue)               # indices of specific bvalue
        self.bval_arr = self.compute_bval_arr(scan, bvlocs)     # dims: (len(bvlocs), 6)
        # signal log is computed differently for DTI1000 and shellDTI
        self.signal_log = self.compute_signal_log(scan, bvlocs, is1000) # dims: (num_of_vols, len(bvlocs))


    def main(self) -> DTI_Values:
        ''' Calculate DTI to each voxel separatelly '''
        ''' Parallellism depends on hardcoded value of num_of_processed '''
        ''' Output: struct with dti values (not a shared one in any case) '''
        if NUM_OF_PROCESSES > 1:
            # run parallelly
            with MyManager() as manager:
                # create a shared object to use in mutliprocessing
                self.values = manager.DTI_Values(self.num_of_vols)
                with Pool(NUM_OF_PROCESSES) as p:   # run volume_dti parallelly
                    p.map(self.volume_dti, range(self.num_of_vols))
                self.extract_data() # copy data from shared object to regular object
        else:
            # run sequentially
            for i in range(self.num_of_vols):
                self.volume_dti(i)

        return self.values
    

    def volume_dti(self, i: int) -> None:
        ''' calculate dti measurements for the i'th voxel '''
        self.volume_dt_calc(i)
        self.volume_eigen_calc(i)
        self.volume_md_calc(i)
        self.volume_fa_calc(i)


    def volume_dt_calc(self, i: int) -> None:
        ''' calculate diffusion tensor for the i'th voxel '''
        ''' diffusion tensor is a 3X3 symmetric matrix '''
        bval_arr = self.bval_arr                            # dims: (n, 6)
        signal_log_i = -1 * (self.signal_log[i, :])         # dims: (n, )
        dt_i = np.linalg.lstsq(bval_arr, signal_log_i, rcond=None)[0]  # dims: (6,)
        self.values.set('dt', array_to_sym(dt_i), i)        # dims: (3,3)


    def volume_md_calc(self, i: int) -> None:
        ''' calculate mean diffusivity for the i'th voxel '''
        ''' mean diffusivity is a float'''
        md_i = np.mean(self.values.get('eigval', i))
        self.values.set('md', md_i, i)


    def volume_fa_calc(self, i: int) -> None:
        ''' calculate fractional anisotropy for the i'th voxel '''
        eigval_i = self.values.get('eigval', i)
        md_i = self.values.get('md', i)
        fa_i = np.sqrt(1.5) * np.linalg.norm(eigval_i - md_i) / np.linalg.norm(eigval_i)
        self.values.set('fa', fa_i, i)


    def volume_eigen_calc(self, i: int) -> None:
        ''' calculate eigen values and eigen vectors
            of the diffusion tensor of the i'th voxel '''
        dt_mat = self.values.get('dt', i)
        eigen_vals, eigen_vecs = np.linalg.eig(dt_mat)  # compute eigen values and vectors
        index = np.argsort(eigen_vals)                  # get indices of sorted eigen values
        eigen_vals = eigen_vals[index] * 1000
        self.volume_eigval_calc(eigen_vals, i)
        self.volume_eigvec_calc(eigen_vecs, eigen_vals, index, i)


    def volume_eigval_calc(self, eigen_vals: np.ndarray, i: int) -> None:
        ''' calculate array of len=3 of eigen values '''
        if np.all(eigen_vals < 0):          # if all eigen values are negative:
            eigen_vals = np.abs(eigen_vals) # change the sign of all values
        eigen_vals[eigen_vals <= 0] = EPS   # replace negative eigen values with EPS
        self.values.set('eigval', eigen_vals, i)


    def volume_eigvec_calc(self, eigen_vecs: np.ndarray, eigen_vals: np.ndarray, index: int, i: int) -> None:
        ''' sort eigen vectors and keep the "last" one 
            based on eigen values increasing order'''
        eigen_vecs = eigen_vecs[:, index]               # sort based in eigen values order
        eigen_vecs = eigen_vecs[:,-1] * eigen_vals[-1]  # take the last vector and multiply it
        self.values.set('eigvec', eigen_vecs, i)

    
    def extract_data(self) -> None:
        ''' copy data from shared object to a new regular object
            set the new object instead of the shared one '''
        shared_values = self.values
        output_values = DTI_Values(self.num_of_vols)    # init new object
        DTI_Values.copy_values(shared_values, output_values, KEYS)  # copy from shared to new
        self.values = output_values # replace it


    def compute_bval_arr(self, scan: Scan, bvlocs: np.ndarray) -> np.ndarray:
        ''' compute in advance for tensors calculation '''
        ''' Output: array of shape (n, 6) '''
        n = len(bvlocs)
        bval_arr = np.zeros((n, 6))
        bval_real = scan.get_bval_data()[bvlocs]      # dims: (n, )
        norm_bvec = scan.get_bvec_norm()[bvlocs, :]   # dims: (n, 3)

        for i in range(n):
            bmat = bval_real[i] * np.outer(norm_bvec[i,:], norm_bvec[i,:])  # dims: (3,3)
            bval_arr[i, :] = sym_mat_to_array(bmat)                         # dims: (6, )
        
        return bval_arr  # dims: (n, 6)
    

    def compute_signal_log(self, scan: Scan, bvlocs: np.ndarray, is1000: bool) -> np.ndarray:

        signal = scan.get_smoothed_data(col_index=bvlocs)         # dims (len(index), n)
        signal_0 = scan.get_signal_0(is1000)[np.newaxis].T        # dims (len(index), 1)
        
        signal_log = np.log((signal / signal_0) + EPS)            # dims: (len(index, n))
        signal_log[np.isnan(signal_log)] = 0
        signal_log[np.isinf(signal_log)] = 0
        
        return signal_log


    def save_dti_files(self, scan: Scan, subj_folder: Path) -> None:
        ''' save data of each measurement to file '''
        files_dict = self.values.get_dict(KEYS)     # create dict of keys and data
        scan.save_files(files_dict, subj_folder)    # use scan object to save data in dict


def dti(scan: Scan, bvalue: float, is1000:bool=False, subj_folder:Path=None) -> DTI_Values:
    ''' envelope function for running AxSI DTI of specific bvalue '''
    ''' is1000 - flag to distinguish DTI1000 and shellDTI '''
    ''' subj_folder - data will be saved to files iff path is given '''
    ''' Output: object containing data calculated in DTI '''
    bvalue_dti = AXSI_DTI(scan, bvalue, is1000)     # init calc object
    # call main function of calc object
    values = bvalue_dti.main()                      # return value is a "result" object
    if subj_folder:                                 # if path was given
        bvalue_dti.save_dti_files(scan, subj_folder)    # save files to path
    return values