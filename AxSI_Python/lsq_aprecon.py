# This file is part of alternative least squares calculation
import numpy as np

EPS = np.finfo('float').eps
EMPTY_ARRAY = np.array([])
import scipy.sparse as sps


class Aprecon():
    def __init__(self, H, DM, DG, n):
        self.H = H
        self.DM = DM
        self.DG = DG
        self.n = n
        self.ppvec = EMPTY_ARRAY
        self.RPCMTX = EMPTY_ARRAY
        # Form matrix m
        self.TM = np.dot(self.H, self.DM)

    def determine_factor(self, pcflags):
        ''' In our case pcflags is preset to inf'''
        # Determin factor of preconditioner
        if not pcflags:    # Diag
            self.diag_preconditioner_factor()
        elif pcflags >= self.n - 1:
            self.singularity_preconditioner_factor()
        else:
            print('aprecon: InvalidUpperBandw')


    def diag_preconditioner_factor(self):
        TM = self.TM
        M = np.dot(TM.T, TM) + self.DG
        dnrms = np.sqrt(np.sum(M * M)).T
        d = max(np.sqrt(dnrms), np.sqrt(EPS))
        self.RPCMTX = d
        self.ppvec = np.arange(self.n)


    def singularity_preconditioner_factor(self):
        TM = self.TM
        dgg = np.sqrt(self.DG)
        TM = np.vstack([TM, dgg])
        self.ppvec = np.arange(2)
        _, self.RPCMTX = np.linalg.qr(TM[:,self.ppvec])
        self.modify_for_singularity(dgg)

    
    def modify_for_singularity(self, dgg):
        mdiag = min(abs(np.diag(self.RPCMTX)))
        lambd = 1
        while mdiag < np.sqrt(EPS):
            TM = np.concatenate(np.dot(self.H, self.DM), dgg + lambd*np.eye(self.n), axis=1)
            self.ppvec = sps.linalg.splu(TM, perm_spec='COLAMD')
            self.RPCMTX = np.linalg.qr(TM[:,self.ppvec])
            lambd = 4 * lambd
            mdiag = min(abs(np.diag(self.RPCMTX)))
