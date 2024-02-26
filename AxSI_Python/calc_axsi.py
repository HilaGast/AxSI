# The main file for AxSI measures calculations

import numpy as np
import cvxpy as cp
from toolbox import MyManager, Values
from toolbox import NUM_OF_PROCESSES, ADD_VALS, N_SPEC
from toolbox import init_l_matrix, init_yd
from matrix_operations import exp_vec_mat_multi
from mri_scan import Scan
from predictions import Predictions
from pathlib import Path
from multiprocessing import Process
from lsq_for_axsi import alternative_least_squares
from predictions import predict
from ax_dti import DTI_Values


# attributes of DTI calculations
KEYS = ['pfr', 'ph', 'pcsf', 'pasi', 'paxsi', 'CMDfr', 'CMDfh']
# least squares parameters
LOWER_BOUND = np.zeros(N_SPEC + 2)
UPPER_BOUND= np.ones(N_SPEC + 2)
X0 = np.asarray([0.5, 5000])
MIN_VAL = np.zeros(2)
MAX_VAL = np.asarray([1, 20000])


class Calc_AxSI_Values(Values):
    ''' An extended class for Values with 
        relevant attributes for AxSI '''
    def __init__(self, n) -> None:
        self.pfr = np.zeros(n)
        self.ph = np.zeros(n)
        self.pcsf = np.zeros(n)
        self.pasi = np.zeros(n)
        self.paxsi = np.zeros((n, N_SPEC))  # N_SPEC is defined in toolbox
        self.CMDfr = np.zeros(n)
        self.CMDfh = np.zeros((n, 2))


# register class to MyManager, used for multiprocessing
MyManager.register('Calc_AxSI_Values', Calc_AxSI_Values)


class Calc_AxSI():
    ''' The main object for performing AxSI analysis '''
    YD = init_yd()                      # static field, init once and use for all iterations
    L_MATRIX = init_l_matrix(N_SPEC)    # static field, init once and use for all iterations

    def __init__(self, scan: Scan, decays: Predictions) -> None:
        ''' use Scan for init, don't keep it '''
        self.decays = decays                        # Predictions object 
        self.data = scan.get_smoothed_data()        # smoothed MRI signal
        self.n_vols = scan.get_num_of_voxels()      # number of brain voxels in mask
        self.values = Calc_AxSI_Values(self.n_vols) # init struct with zero values
        # compute in advance
        self.vCSF = init_pixpredictCSF(scan.get_grad_dirs())    # depends on scan values, dims: (n_frames, )

    
    class Block:
        ''' Object for a singel itaration data '''
        def __init__(self, decays: Predictions, ydata: np.ndarray, vCSF: np.ndarray, i: int) -> None:
            ''' use decays to predict arrays for single voxel only '''
            self.i = i
            self.ydata = ydata                         # scan signal, dims: (n_frames, )
            self.vCSF = vCSF                           # dims: (n_frames, )
            self.vH = decays.get_hindered_slice(i)     # dims: (n_frames, )
            self.vR = decays.get_restricted_slice(i)   # dims: (n_frames, N_SPEC)
            self.prcsf_i = decays.get_csf_slice(i)     # float
            self.vH[self.vH > 1] = 0                   # correct vH data
        
        def stack(self) -> np.ndarray:
            ''' put vR, vH, vCSF in matrix together '''
            vR = np.nan_to_num(self.vR, nan=0)                  # replace NaN with 0
            preds = np.column_stack((vR, self.vH, self.vCSF))   # dims: (n_frames, N_SPEC+2)
            return preds


    # The same as in DTI file
    def save_calc_files(self, scan: Scan, subj_folder: Path) -> None:
        ''' save data of each measurement to file '''
        files_dict = self.values.get_dict(KEYS)     # create dict of keys and data
        scan.save_files(files_dict, subj_folder)    # use scan object to save data in dict


    def main(self) -> None:
        ''' Calculate AxSI to each voxel separatelly '''
        ''' Parallellism depends on hardcoded value of num_of_processed '''
        ''' Output: struct with AxSI values (not a shared one in any case) '''
        if NUM_OF_PROCESSES > 1:
            # run in parallel
            self.parallel_calc_axsi()
        else:
            # run sequentially
            for i in range(self.n_vols):
                Calc_AxSI.perform_iteration(self.values, self.decays, self.data[i], self.vCSF, i)

        self.values.nonzero('pfr')
        self.values.set('pcsf', self.decays.csf)


    def parallel_calc_axsi(self) -> None:
        ''' Calculate AxSI to each voxel in parallel '''
        ''' Use multiprocessing and Manager to write data to the same object '''
        ''' Data will be stored in object '''
        # use Manager to share data between processes
        with MyManager() as manager:
            # create a shared object
            shared_values = manager.Calc_AxSI_Values(self.n_vols)
            # init empty array for processes
            procs = []
            # for each voxel
            for i in range(self.n_vols):
                # create a process to calculate voxel attributes
                proc = Process(target=Calc_AxSI.perform_iteration, 
                                args=(shared_values, self.decays, self.data[i], self.vCSF, i))
                procs.append(proc)  # append process to array
                proc.start()        # start process
            # wait for all processes to finish
            for proc in procs:
                proc.join()
            # copy values from a shared object to self
            Values.copy_values(shared_values, self.values, KEYS)
    

    def perform_iteration(values: Values, decays: Predictions, ydata: np.ndarray, vCSF: np.ndarray, i: int) -> None:
        ''' calculate AxSI measurements for the i'th voxel '''
        if i // 1000 == i / 1000:
            print(f'calc axsi iter: {i}')
        # init an object with iteration data 
        block = Calc_AxSI.Block(decays, ydata, vCSF, i)
        # nonlinear least squares with predictions and signal
        parameter_hat = Calc_AxSI.least_squares_envelope(block)
        # linear least squares with predictions and signal
        x = Calc_AxSI.solve_vdata(block, parameter_hat)
        # update iteration results in object
        Calc_AxSI.set_values(values, x, parameter_hat, block)


    def least_squares_envelope(block: Block) -> np.ndarray:
        ''' Use own version to achieve more similar results to MATLAB lsqnonlin '''
        ''' Input: object with iteration data '''
        ''' Output: array with two elements '''
        vRes = np.dot(block.vR, Calc_AxSI.YD)   # YD is the same for all voxels, dims: (n_frames, )
        vRes = np.nan_to_num(vRes, nan=0)       # replace NaN with 0
        # run nonlinear least squares with regression function 
        parameter_hat = alternative_least_squares(reg_func, X0, 
                                                  bounds=(MIN_VAL, MAX_VAL), ftol=1e-6, xtol=1e-6, 
                                                  diff_step=1e-3, jac=jac_calc, max_nfev=20000, 
                                                  args=(block.ydata, block.vH, vRes, block.vCSF, block.prcsf_i)).x
        return parameter_hat
    

    def solve_vdata(block: Block, parameter_hat: np.ndarray) -> np.ndarray:
        ''' linear least squares with predictions and signal '''
        ''' Input: iteration data and results of lsqnonlin '''
        ''' Output: np array of shape (N_SPEC+2, )'''
        # divide signal with nonlinlsq result
        vdata = block.ydata / parameter_hat[1]          # dims: (n_frames, )
        # adjust variables to current voxels
        LOWER_BOUND[-1] = block.prcsf_i - 0.02          # change the last element
        UPPER_BOUND[-1] = block.prcsf_i + 0.02          # change the last element
        Xprim = np.concatenate((block.stack(), Calc_AxSI.L_MATRIX)) # dims: (n_frames+N_SPEC, N_SPEC+2)
        Yprim = np.concatenate((vdata, np.zeros(160)))              # dims: (n_frames+N_SPEC, )
        # run linear least squares
        x = lin_least_squares_with_constraints(Xprim, Yprim, LOWER_BOUND, UPPER_BOUND)    # dims: (N_SPEC+2, )
        return x
    

    def set_values(values: Calc_AxSI_Values, x: np.ndarray, parameter_hat: np.ndarray, block: Block) -> None:
        ''' update object with current voxel values '''
        i = block.i                             # voxel index
        x[x < 0] = 0                            # replace negative values with zeros
        nx = x[:130]                            # take the first 130 element
        nx = nx / np.sum(nx)                    # normalize
        pasi_i = np.sum(nx * ADD_VALS[:130])    # ADD_VALS is defined in toolbox

        values.set('ph', x[160], i)
        values.set('pfr', 1 - block.prcsf_i - x[160], i)
        values.set('pasi', pasi_i, i)
        values.set('paxsi', x[:160], i)
        values.set('CMDfh', parameter_hat, i)
        values.set('CMDfr', 1 - parameter_hat[0] - block.prcsf_i, i)
    


def calc_axsi(scan: Scan, dti1000: DTI_Values, shell_dti: list[DTI_Values], subj_folder: Path):
    ''' envelope function for running AxSI '''
    ''' use DTI1000 and shellDTI calculated for scan '''
    ''' subj_folder - data will be saved to files in path '''
    decays = predict(scan, dti1000, shell_dti)  # Calculate predictions decays for each environment
    # init AxSI object
    calc = Calc_AxSI(scan, decays)
    # run main function of AxSI analysis
    calc.main()
    print("Saving calc files")
    calc.save_calc_files(scan, subj_folder)


def init_pixpredictCSF(grad_dirs: np.ndarray):
    ''' a series of matrix multiplications '''
    ''' Input: matrix of mXn '''
    ''' Output: np array of shape (m, ) '''
    D_mat = np.eye(grad_dirs.shape[1]) * 4
    pixpredictCSF = exp_vec_mat_multi(grad_dirs, D_mat, -4)
    return pixpredictCSF


def reg_func(x, ydata, pixpredictH, pixpredictR, pixpredictCSF, prcsf):
    ''' regression function for nonlinear least squares '''
    ''' xt = 1 - x[0] - prcsf
    newdata = x[1] * (x[0] * pixpredictH + xt * pixpredictR + prcsf * pixpredictCSF) '''
    newdata = x[1] * (x[0] * pixpredictH + (1-x[0]-prcsf) * pixpredictR + prcsf * pixpredictCSF)
    err = newdata - ydata

    return err


def jac_calc(x, ydata, pixpredictH, pixpredictR, pixpredictCSF,prcsf):
    ''' jacobian matrix calculation for nonlinear least squares '''
    jac = np.zeros([len(ydata),2])
    jac[:,0] = x[1] * (pixpredictH - pixpredictR)
    jac[:,1] = x[0]*pixpredictH + (1-x[0]-prcsf) * pixpredictR + prcsf * pixpredictCSF

    return jac


def lin_least_squares_with_constraints(A, b, lb, ub):
    ''' use cvxpy to find x such that :
        minimize sum of squares for [A @ x - b] 
        lb <= x <= ub and sum(x) == 1 '''
    # Define your variables and problem data
    n = A.shape[1]
    x = cp.Variable(n)
    objective = cp.Minimize(0.5 * cp.sum_squares(A @ x - b))
    constraints = [x >= lb, x <= ub, cp.sum(x) == 1]

    # Create the optimization problem
    prob = cp.Problem(objective, constraints)

    # Try to solve the problem
    try:
        prob.solve(warm_start=True, solver=cp.ECOS)
        # Optionally, you can check the status and the optimal value:
    except cp.SolverError:
        print("SolverError: The problem does not have a feasible solution.")
        return np.zeros(x.shape)

    return x.value
