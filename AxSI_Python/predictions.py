# This file is for computing Prediction decays for AxSI
# predict CSF and predicit_hindered are computed at once in advance
# predict restricted takes to much memory so prepare to compute for each voxel later

import numpy as np
from mri_scan import Scan
from ax_dti import DTI_Values
from multiprocessing import Pool
from scipy.optimize import curve_fit
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from matrix_operations import exp_vec_mat_multi
from toolbox import cart2sph_env, NUM_OF_PROCESSES, ADD_VALS


class Predictions():
    ''' Main class for predictions '''
    def __init__(self, scan: Scan, dti1000: DTI_Values) -> None:
        ''' use scan for init, don't keep it '''
        self.num_of_vols = scan.get_num_of_voxels() # number of brain voxels in mask
        # init empty arrays for predict_hindered and predict_csf
        self.md0 = np.zeros(self.num_of_vols)
        self.hindered = np.zeros((self.num_of_vols, scan.get_bval_length()))
        self.csf = np.zeros(self.num_of_vols)
        # keep only necessary data from scan and DTI
        self.grad_dirs = scan.get_grad_dirs()   # used for predict_hindered
        self.max_bval = scan.get_max_bval()     # used for predict_hindered
        self.bshell = scan.get_shell()          # used for MD0 curve fit
        self.params = scan.get_params()         # used for predict_restricted init
        self.signal_0 = scan.get_signal_0()     # used for predict_restricted
        self.eigvec1000 = dti1000.get("eigvec") # used for predict_restricted_init
        self.dt1000_maps = dti1000.get("dt")    # used for predict_hindered


    def get_hindered_slice(self, i) -> np.ndarray:
        ''' get the ith entry of precomputed decay '''
        return self.hindered[i, :]
    
    def get_restricted_slice(self, i) -> np.ndarray:
        ''' not precomputed, run predict_restricted '''
        return self.predict_restricted(i)
    
    def get_csf_slice(self, i) -> float:
        ''' get the ith entry of precomputed decay '''
        return self.csf[i]


    def predict_hindered(self, bshell: np.ndarray, shell_dti: list[DTI_Values]) -> None:
        ''' Main function to compute predict hindered and MD0 for each voxel '''
        ''' Parallellism depends on hardcoded value of num_of_processed '''
        # sMDi[j,k] is the biggest eigval of voxel j under the k'th DTI, dims: (num_of_vols, len(bshell))
        sMDi = np.stack([dti.get("eigval")[:,-1] for dti in shell_dti], axis=-1)    

        if NUM_OF_PROCESSES > 1:
            # use multiprocess pool to calculate MD0 in parallel
            with Pool(NUM_OF_PROCESSES) as p:
                # starmap returns array of results of all processes
                md0 = p.starmap(Predictions.predict_md0_for_voxel, 
                                [(bshell, sMDi[i, :]) for i in range(self.num_of_vols)])
                self.md0 = np.array(md0)    # convert array to numpy
        else:
            # calculate MD0 in sequence
            for i in range(self.num_of_vols):
                self.md0[i] = Predictions.predict_md0_for_voxel(bshell, sMDi[i, :])
        # calculate hindered decay altogether
        self.predict_hindered_decay(sMDi)


    def predict_hindered_decay(self, sMDi: np.ndarray) -> None:
        ''' calculate hindered decay for all voxels at once '''
        # divide MD0 with eigvals of shell DTI with bvalue=1
        every_fac = self.md0 / sMDi[:, 0]
        every_fac = every_fac.reshape((self.num_of_vols, 1, 1)) # adjust to DT1000_maps dims
        every_d_mat = every_fac * 1000 * self.dt1000_maps       # dims: (num_of_vols, 3, 3)
        # a series of matrix multiplications 
        decay = exp_vec_mat_multi(self.grad_dirs, every_d_mat, -self.max_bval, axis=1)
        self.hindered = decay   # set result


    def predict_md0_for_voxel(bshell: np.ndarray, smdi_i: np.ndarray) -> float:
        ''' for a specific voxel, run curve fit 
            with bshell values and eigvals (of DTI with respective bshell) '''
        fit_a = curve_fit(exp1, bshell, smdi_i, method='trf', loss='soft_l1')[0][0]    # not precisely as in matlab
        return fit_a
    

    def init_predict_restricted(self) -> None:
        ''' Cannot calculate restricted decay in advance (memory limitations) '''
        ''' compute partially and avoid repeated calculations '''
        factor_angle_term_par = self.calc_factor_angle()            # dims: (n_vols, n_frames)
        factor_angle_term_perp = np.sqrt(1 - factor_angle_term_par**2)  # dims: (n_vols, n_frames)
        self.q_par_sq_arr = self.calc_sq(factor_angle_term_par)     # dims: (n_vols, n_frames)
        self.q_perp_sq_arr = self.calc_sq(factor_angle_term_perp)   # dims: (n_vols, n_frames)


    def predict_restricted(self, i: int) -> np.ndarray:
        ''' predict restricted decay for the ith voxel '''
        ''' use data computed to all voxels '''
        ''' output dims: (n_frames, N_SPEC) '''
        signal_0 = self.signal_0[i]             # mean of voxel i signal with bvalue=0  
        q_par_sq_arr = self.q_par_sq_arr[i, :]
        q_perp_sq_arr = self.q_perp_sq_arr[i, :]
        md = self.md0[i]                        # MD0 from predict_hindered calculations

        E = self.calc_E(q_perp_sq_arr, q_par_sq_arr, md)    # calculate E matrix, dims: (n_frames, N_SPEC)
        decay = signal_0 * E                                # consider signal
        self.restricted = decay / np.nanmax(decay)          # normalize with max value

        return self.restricted
    

    def calc_factor_angle(self) -> np.ndarray:
        '''compute for all eigvec at once'''
        ''' eigvecs: (a, ), factor_angle: (a, n_frames) - a is num_of_vols '''
        theta_q = self.params['theta']                          # dims: (n_frames, )
        phi_q = self.params['phi']                              # dims: (n_frames, )
        _, phi_n, theta_n = cart2sph_env(self.eigvec1000.T)     # dims: (a, )
        tmp_cos = np.outer(np.cos(theta_n), np.cos(theta_q))    # dims: (a, n_frames)
        tmp_sin = np.outer(np.sin(theta_n), np.sin(theta_q))    # dims: (a, n_frames)
        factor_angle = abs(tmp_cos * np.cos(phi_q - phi_n[np.newaxis].T) + tmp_sin) # dims: (a, n_frames)
        return factor_angle


    def calc_sq(self, factor_angle: np.ndarray) -> np.ndarray:
        ''' factor_angle: (a, n_frames)'''
        return (self.params['r'] * factor_angle) ** 2


    def calc_E(self, q_perp: np.ndarray, q_par: np.ndarray, MD: float) -> np.ndarray:
        ''' Calculate E matrix for predict restricted '''
        '''qprep: (n_frames, ), qpar: (n_frames, )'''
        bigdel = self.params['big_delta']
        smldel = self.params['small_delta']
        q_par = q_par[np.newaxis].T                         # adjust dims to (n_frames, 1)
        # ADD_VALS is defined in toolbox
        tmp = np.multiply.outer(q_perp, (ADD_VALS/2)**2)    # dims: (n_frames, N_SPEC)
        tmp = tmp + (q_par * (bigdel - smldel / 3) * MD)    # dims: (n_frames, N_SPEC)
        E = np.exp(-4 * np.pi**2 * tmp)                     # dims: (n_frames, N_SPEC)
        return E
    

    def predict_csf(self, shell_dti: list[DTI_Values]) -> np.ndarray:
        ''' calculate CSF decay for all voxels at once '''
        ''' use gaussian mixture model '''
        md_non0 = shell_dti[0].get("md")            # get md values from shellDTI with bvalue=0
        md_non0 = md_non0[md_non0 > 0]              # take only positive values
        # fit data
        gm = self.fit_gaussian_mixture(md_non0)
        p = self.compute_pdf_array(md_non0, gm)     # array of pdf

        self.csf = p[2] / (p[0] + p[1] + p[2])
        self.csf[np.isnan(self.csf)] = 0

        return self.csf


    def compute_pdf_array(self, md_nonzero: np.ndarray, gm: GaussianMixture) -> list:
        comp_mu = gm.means_             # means of distribution of model
        comp_sigma = gm.covariances_    # covariances of distribution of model
        # compute pdf for each mu and sigma
        p0 = norm.pdf(md_nonzero, comp_mu[0,0], np.sqrt(comp_sigma[0,0]))
        p1 = norm.pdf(md_nonzero, comp_mu[1,0], np.sqrt(comp_sigma[1,0]))
        p2 = norm.pdf(md_nonzero, comp_mu[2,0], np.sqrt(comp_sigma[2,0]))

        return [p0, p1, p2]


    def fit_gaussian_mixture(self, md_nonzero: np.ndarray):
        ''' use scipy gaussian mixture fit data '''
        md_nonzero = md_nonzero.reshape(-1, 1)
        sigma = np.full((3,1,1), 0.3 * np.nanmax(md_nonzero))
        params = {'n_comp': 3, 
                'mu': np.asarray([0.5,1,2]).reshape((3,1)), 
                'weights': np.array([0.7,0.2,0.1]),
                'sigma': sigma, 
                'tol': 1e-9, 
                'max_iter': 10000,
                'init_params': 'random'
                }
        gm = GaussianMixture(
            n_components=params['n_comp'], covariance_type='full', tol=params['tol'], 
            reg_covar=1e-7, max_iter=params['max_iter'], init_params=params['init_params'], 
            weights_init=params['weights'], means_init=params['mu'], precisions_init=params['sigma']
        ).fit(md_nonzero)    # not precisely as in matlab
        return gm
    


def exp1(x,a,b):
    ''' function for curve fit in MD0 calculation'''
    y = a*np.exp(b*x)
    return y


def predict(scan: Scan, dti1000: DTI_Values, shell_dti: list[DTI_Values]):
    ''' Input: scan object, DTI1000 and shellDTI '''
    ''' Output: object with decays prediction for AxSI'''
    decays = Predictions(scan, dti1000)     # init object
    # dims hindered_decay: (num_of_vols, timeframes)   dims md0: (num_of_vols, )
    decays.predict_hindered(scan.get_shell(), shell_dti)
    # predict restricted cannot be calculated in advance because of memory limitations
    decays.init_predict_restricted()
    # calculate CSF decay, dims: (num_of_vols)
    decays.predict_csf(shell_dti)

    return decays