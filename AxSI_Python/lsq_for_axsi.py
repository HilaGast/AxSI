# This file serves as an equivalent to MATLAB lsqnonlin
# accuracy and efficiency need improvement

import numpy as np
from numpy.linalg import norm
from lsq_components import *
from lsq_trust_region import Trust_region


EPS = np.finfo('float').eps


class LSQ():
    def __init__(self, reg_func, xstart, user_options, jacob, *varargin):
        self.func = reg_func
        self.options = LSQOptions(user_options, xstart)
        self.values = LSQValues(reg_func, xstart, jacob, *varargin)
        self.vars = LSQVars(self.values, self.options)
        self.vars.increment_param("numFunEvals", 3)
        self.result = LSQResult(xstart, self.values.fvec, self.values.A)
        self.tregion = None
    

    def set_boundaries(self, lower_bound, upper_bound):
        self.vars.set_lower_bound(lower_bound)
        self.vars.set_upper_bound(upper_bound)
        self.vars.assert_bounds()
        self.assert_x_in_bounds()


    def assert_x_in_bounds(self):
        x = self.values.x
        ub = self.vars.ub
        lb = self.vars.lb
        if min(min(ub - x), min(x - lb)) < 0:
            self.values.x = self.vars.compute_startx()
    

    def main_loop(self, jac, *varargin):
        while not self.vars.ex:
            self.vars.evaluate_dnewt(self.values.fvec)
            self.update_iteration_variables()
            self.test_for_convergence()
            if not self.vars.ex:
                self.perform_iteration(jac, *varargin)
            self.check_iterations_limit()        

    
    def perform_iteration(self, jac, *varargin):
        reg_func = self.func
        
        newx = self.determine_trust_region_correction()
        newx =  self.perturb(newx)
        new_values = LSQValues(reg_func, newx, jac, *varargin)
        self.vars.increment_param("numFunEvals", 3)
        self.test_trial_point(new_values.val)
        self.advance_to_next_iteration(new_values)


    def test_trial_point(self, new_val):
        self.update_ratio(new_val)
        self.update_delta(new_val)


    def update_ratio(self, new_val):
        ss = self.tregion.ss
        dv = self.vars.dv
        g = self.values.grad
        val = self.values.val
        qp = self.tregion.qpval
        aug = 0.5 * np.dot(ss.T, (dv * abs(g)) * ss)
        self.vars.ratio = (0.5 * (new_val - val) + aug) / qp

    
    def update_delta(self, new_val):
        delta = self.vars.delta
        ratio = self.vars.ratio
        nrmsx = self.vars.nrmsx
        
        if ratio >= 0.75 and nrmsx >= 0.9*delta:
            self.vars.delta = min(self.vars.delbnd, 2*delta)
        elif ratio <= 0.25:
            self.vars.delta = min(nrmsx / 4, delta / 4)
        if np.isinf(new_val):
            self.vars.delta = min(nrmsx / 20, delta / 20)


    def advance_to_next_iteration(self, new_values):
        if new_values.val < self.values.val:
            self.values = new_values
        self.vars.it = self.vars.it + 1
        self.vars.vval[self.vars.it] = self.values.val


    def perturb(self, x, delta=100*EPS):
        u = self.vars.ub
        l = self.vars.lb
        if (min(abs(u - x)) < delta) or (min(abs(x - l)) < delta):
            upperi = (u-x) < delta
            loweri = (x-l) < delta
            x[upperi] = x[upperi] - delta
            x[loweri] = x[loweri] + delta
        return x


    def update_iteration_variables(self):
        self.vars.define_dv_and_v(self.values.x, self.values.grad)
        gopt = self.vars.v * self.values.grad
        self.vars.voptnrm[self.vars.it] = norm(gopt, np.inf)


    def test_for_convergence(self):
        nrmsx = self.vars.nrmsx
        delta = self.vars.delta
        ratio = self.vars.ratio
        tol1 = self.options.tol1
        val = self.values.val
        it = self.vars.it
        if it > 0:
            diff = abs(self.vars.vval[it - 1] - val)
            if (nrmsx < 0.9*delta) & (ratio > 0.25) & (diff < tol1*(1 + abs(val))):
                self.result.successful_termination()
                self.vars.ex = 1
            elif (nrmsx < self.options.tol2):
                self.result.successful_termination()
                self.vars.ex = 2


    def check_iterations_limit(self):
        if self.vars.it > self.options.itb:
            self.max_iterations_exceeded()
        if self.vars.numFunEvals > self.options.maxfunevals:
            self.max_iterations_exceeded()

    
    def max_iterations_exceeded(self):
        self.vars.ex = 4
        self.vars.it = self.vars.it - 1
        self.result.iterations_exceeded()


    def determine_trust_region_correction(self):
        iter = self.vars.it
        optnorm = self.vars.voptnrm[iter]
        theta = max(0.95, 1 - optnorm)
        self.tregion = self.trust_region_trial_step(theta)
        newx = self.update_tregion_values()
        return newx


    def trust_region_trial_step(self, theta):
        x = self.values.x
        delta = self.vars.delta
        options = self.options

        tregion = Trust_region(self.values, self.vars)
        tregion.trial_step(x, theta, options.kmax, options.pcflags, options.pcgtol, delta)
        return tregion


    def update_tregion_values(self):
        iter = self.vars.it
        posdef = self.tregion.posdef
        if posdef == 0:
            posdef = self.vars.vpos[iter]
        self.vars.vpos[iter+1] = posdef
        self.vars.nrmsx = norm(self.tregion.ss)
        self.vars.vpcg[iter+1] = self.tregion.pcgit
        newx = self.values.x + self.tregion.s
        return newx

                
    def prepare_output(self):
        self.result.jacob = self.values.A
        self.set_output_values()
        self.result.x = self.values.x
        self.result.fvec = self.values.fvec
        

    def set_output_values(self):
        it = self.vars.it
        output = self.result.output
        output['firstorderopt'] = self.vars.voptnrm[it]
        output['iterations'] = it
        output['funcCount'] = self.vars.numFunEvals
        output['cgiterations'] = np.sum(self.vars.vpcg)
        


def snls(funfcn, xstart, l, u, options, jac, *varargin):

    lsq_component = LSQ(funfcn, xstart, options, jac, *varargin)
    lsq_component.set_boundaries(l,u)
    lsq_component.main_loop(jac, *varargin)
    lsq_component.prepare_output()

    return lsq_component.result
    
    
def alternative_least_squares(reg_func, x0, bounds=None, ftol=1e-6, xtol=1e-6, diff_step=1e-3, jac=None, max_nfev=20000, args=None):

    xstart = np.array(x0, dtype='float')
    l = np.array(bounds[0], dtype='float')
    u = np.array(bounds[1], dtype='float')
    options = {'TolFun': ftol, 'TolX':xtol, 'MaxFunEvals': max_nfev, 'diff_step': diff_step}
    varargin = []
    for i in range(len(args) - 1):
        varargin.append(np.array(args[i]).flatten())
    varargin.append(args[-1])

    result = snls(reg_func, xstart, l, u, options, jac, *varargin)
    return result
