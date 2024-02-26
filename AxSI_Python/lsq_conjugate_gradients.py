# This file is part of alternative least squares calculation

import numpy as np
from numpy.linalg import norm
from matrix_operations import *

class Precondition():
    def __init__(self, H, DM, DG, grad):
        self.H = H
        self.DM = DM
        self.DG = DG
        self.r = -grad
        self.rshape = self.r.shape
        self.v1 = np.zeros(self.rshape)
        self.it = 0
        self.posdef = 1
        self.inner1 = 0
        self.inner2 = 0
        

    def init_procedure(self, ppvec, RPCMTX, tol):
        self.compute_vars(0, ppvec, RPCMTX)
        self.stoptol = tol * norm(self.gradient_z)


    def solve(self, kmax, ppvec, RPCMTX, tol):
        self.init_procedure(ppvec, RPCMTX, tol)

        # Primary loop
        for k in range(max(kmax, 1)):
            if self.conjugate_gradients_break():
                break
            self.conjugate_gradients_advance(k, ppvec, RPCMTX)
            # Exit?
            if norm(self.gradient_z) <= self.stoptol:
                break
        
        self.v1 = normalize(self.v1)
        self.it = k

    
    def conjugate_gradients_break(self):
        if self.denom <= 0:
            self.posdef = 0
            self.v1 = self.d
            return True
        return False

    
    def conjugate_gradients_advance(self, k, ppvec, RPCMTX):
        alpha = self.inner1 / self.denom
        self.v1 = self.v1 + alpha * self.d
        self.r = self.r - alpha * self.ww
        self.compute_vars(k, ppvec, RPCMTX)


    def compute_vars(self, k, ppvec, RPCMTX):
        self.gradient_z = self.apply_preconditioner_to_vector(ppvec, RPCMTX)
        self.compute_d(k)
        w = compute_w(self.H, self.DM, self.d)
        self.ww = w + np.dot(self.DG, self.d)
        self.denom = np.dot(self.d.T, self.ww)
        self.inner2 = self.inner1
        self.inner1 = np.dot(self.r.T, self.gradient_z)


    def compute_d(self, k):
        if k == 0:
            self.d = self.gradient_z
        else:
            beta = self.inner1 / self.inner2
            self.d = self.gradient_z + beta*self.d
    

    def apply_preconditioner_to_vector(self, ppvec, RPCMTX):
        # initialization
        w = np.zeros(self.rshape)
        if ppvec.size == 0:
            ppvec = np.arange(self.rshape[0])
            if RPCMTX.size == 0:
                RPCMTX = np.eye(self.rshape[0])

        # Precondition
        wbar = np.linalg.solve(RPCMTX.T, self.r[ppvec])
        w[ppvec] = np.linalg.solve(RPCMTX, wbar)

        return w