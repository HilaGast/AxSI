# This file is part of alternative least squares calculation

import numpy as np
from numpy.linalg import norm
from matrix_operations import *

EPS = np.finfo('float').eps
EMPTY_ARRAY = np.array([])
INF = np.inf
defaultopt = {
    'ActiveConstrTol': np.sqrt(EPS),
    'PrecondBandWidth': INF,
    'tolPCG': 0.1,
    'MaxPCGIter': 'max(1,floor(numberofvariables/2))',
    "TypicalX": 'ones(numberofvariables,1)',
    "maxfunevals": '100*numberofvariables',
    'MaxIter': 20000
}


class LSQResult():
    def __init__(self, xstart, fvec, jacob):
        self.x = xstart
        self.fvec = fvec
        self.jacob = jacob
        self.exitflag = 0
        self.output = {'algorithm':'trust-region-reflective'}
        self.msg = ""

    def successful_termination(self):
        self.exitflag = 1
        self.msg = "Optimization terminated successfully"
    
    def iterations_exceeded(self):
        self.exitflag = 0
        self.msg = "Maximum number of iterations exceeded"




class LSQOptions():
    def __init__(self, user_options, xstart):
        self.user_options = user_options
        self.default_options = defaultopt
        self.active_tol = self.optimget('ActiveConstrTol')
        self.gradflag = self.optimget('Jacobian') == 'on'
        self.typx = self.optimget('TypicalX')
        self.pcflags = self.optimget('PrecondBandWidth')
        self.tol2 = self.optimget('TolX')
        self.tol1 = self.optimget('TolFun')
        self.itb = self.optimget('MaxIter')
        self.maxfunevals = self.optimget('MaxFunEvals')
        self.pcgtol = self.optimget('tolPCG')
        self.kmax = self.optimget('MaxPCGIter')
        self.numberOfVariables = xstart.shape[0]
        self.assert_options()

    def optimget(self, key, default_value=None):
        if key in self.user_options.keys():
            return self.user_options[key]
        if key in self.default_options.keys():
            return self.default_options[key]
        return default_value
    
    def assert_options(self):
        if self.numberOfVariables == 0:
            print("Warning: Number of variables must be positive")
        if self.pcgtol <= 0:
            self.pcgtol = 0.1
        self.assert_attribute("typx", 'ones(numberofvariables,1)', np.ones(self.numberOfVariables))
        self.assert_attribute("kmax", 'max(1,floor(numberofvariables/2))', max(1, self.numberOfVariables // 2))
        self.assert_attribute("maxfunevals", '100*numberofvariables', 100*self.numberOfVariables)
        
    def assert_attribute(self, attribute, pattern, default_value):
        current_value = getattr(self, attribute)
        if isinstance(current_value, str):
            if current_value.lower() == pattern:
                setattr(self, attribute, default_value)
            else:
                print(f"Option {attribute} must be integer value if not the default")
    


class LSQValues():
    def __init__(self, reg_func, x, jac, *varargin):
        self.x = x
        self.fvec = reg_func(x, *varargin)
        self.A = jac(x, *varargin)
        self.grad = atamult(self.A, self.fvec, -1)
        self.val = np.dot(self.fvec, self.fvec)
        self.assert_fvec()

    def assert_fvec(self):
            if self.fvec.shape[0] < self.x.shape[0]:
                print("the number of equations must not be less than n")


class LSQVars():
    def __init__(self, values: LSQValues, options: LSQOptions):
        self.n = values.x.shape[0]    # param
        self.lb = EMPTY_ARRAY   # param
        self.ub = EMPTY_ARRAY   # param
        self.dnewt = EMPTY_ARRAY
        self.it = 0             # counters
        self.numFunEvals = 0    # counters
        self.numGradEvals = 0   # counters
        self.ex = 0
        self.vpcg = np.zeros(options.itb)   # vector of trustregion.pcgit
        self.vpos = np.ones(options.itb)    # vector of posdef
        self.vval = np.zeros(options.itb)   # vector of vals
        self.vval[self.it] = values.val
        self.voptnrm = np.zeros(options.itb)    # vector
        self.delta = 10
        self.nrmsx = 1
        self.ratio = 0
        self.v = EMPTY_ARRAY
        self.dv = EMPTY_ARRAY
        self.delbnd = max(100 * norm(values.x), 1)    # param

    def increment_param(self, name, add):
        setattr(self, name, getattr(self, name) + add)

    def set_lower_bound(self, lower_bound):
        if lower_bound.size == 0:
            lower_bound = -INF * np.ones(self.n)
        lower_bound[lower_bound <= -1e10] = -INF
        self.lb = lower_bound

    def set_upper_bound(self, upper_bound):
        if upper_bound.size == 0:
            upper_bound = INF * np.ones(self.n)
        upper_bound[upper_bound >= 1e10] = INF
        self.ub = upper_bound

    def assert_bounds(self):
        if np.any(self.ub == self.lb):
            print("equal upper and lower bound not permitted")
            exit()
        elif min(self.ub - self.lb) <= 0:
            print("inconsistent bounds")
            exit()


    def compute_startx(self):
        '''startx returns centered point'''
        arg1 = (self.ub <  INF) & (self.lb == -INF)
        arg2 = (self.ub == INF) & (self.lb >  -INF)
        arg3 = (self.ub <  INF) & (self.lb >  -INF)
        arg4 = (self.ub == INF) & (self.lb == -INF)
        return self.set_startx_values(arg1, arg2, arg3, arg4)        


    def set_startx_values(self, arg1, arg2, arg3, arg4):
        xstart = np.zeros(self.n)
        w = np.maximum(abs(self.ub), np.ones(self.n))
        ww = np.maximum(abs(self.lb), np.ones(self.n))

        xstart[arg1] = self.ub[arg1] - 0.5*w[arg1]
        xstart[arg2] = self.lb[arg2] + 0.5*ww[arg2]
        xstart[arg3] = (self.ub[arg3] + self.lb[arg3]) / 2
        xstart[arg4] = 1
        return xstart


    def define_dv_and_v(self, x, grad):     
        arg1 = (grad <  0) & (self.ub <   INF)
        arg2 = (grad >= 0) & (self.lb >  -INF)
        arg3 = (grad <  0) & (self.ub ==  INF)
        arg4 = (grad >= 0) & (self.lb == -INF)

        self.define_v(x, arg1, arg2, arg3, arg4)
        self.define_dv(arg1, arg2, arg3, arg4)


    def define_v(self, x, arg1, arg2, arg3, arg4):
        v = np.zeros(self.n)
        v[arg1] = x[arg1] - self.ub[arg1]
        v[arg2] = x[arg2] - self.lb[arg2]
        v[arg3] = -1
        v[arg4] = 1
        self.v = v


    def define_dv(self, arg1, arg2, arg3, arg4):
        dv = np.zeros(self.n)
        dv[arg1] = 1
        dv[arg2] = 1
        dv[arg3] = 0    # maybe redundant
        dv[arg4] = 0    # maybe redundant
        self.dv = dv


    def evaluate_dnewt(self, fvec):
        if np.any(np.isinf(fvec)):
            print("user function is returning inf or NaN values")
        if fvec.ndim == 2 and fvec.shape[1] == 2:
            self.vars.dnewt = fvec[:,1]
