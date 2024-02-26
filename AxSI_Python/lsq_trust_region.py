# This file is part of alternative least squares calculation

import numpy as np
from numpy.linalg import norm
from matrix_operations import *
from lsq_components import *
from lsq_conjugate_gradients import Precondition
from lsq_aprecon import Aprecon


EPS = np.finfo('float').eps
EMPTY_ARRAY = np.array([])
INF = np.inf



class Trust_region():
    def __init__(self, values:LSQValues, vars: LSQVars):
        self.g = values.grad
        self.H = values.A
        self.ub = vars.ub
        self.lb = vars.lb

        self.DM = np.diag(np.sqrt(abs(vars.v)))
        self.DG = np.diag(abs(self.g) * vars.dv)
        self.grad = np.dot(self.DM, self.g)
        self.n = self.g.shape[0]
        self.Z = EMPTY_ARRAY
        self.v1 = vars.dnewt
        self.pcgit = 0
        self.posdef = 1
        self.exact_trust = Trust_Exact_Solution(vars.delta)


    class QP():
        def __init__(self, st, ss, s):
            self.val = INF
            self.alpha = 1
            self.st = st
            self.ss = ss
            self.s  = s


        def __le__(self, other):
            ''' Compares to instances of QP '''
            return self.val <= other.val


        def multiply_vars(self, coeff=None):
            if coeff == None:
                coeff = self.alpha
            self.s  = coeff * self.s
            self.ss = coeff * self.ss
            self.st = coeff * self.st

        def update_val(self, rhs, MM, addition = 0):
            st = self.st
            val = np.dot(rhs.T, st) + np.dot((0.5 * st).T, np.dot(MM, st))
            self.val = val + addition


    def trial_step(self, x, theta, kmax, pcflags, pcgtol, delta):
        self.determine_subspace(x, theta, kmax, pcflags, pcgtol)
        if len(self.g) > 1:
            self.reflected_direction_evaluation(x, delta, theta)
            self.gradient_direction_evaluation(x, theta)
        self.choose_best_direction()



    def determine_subspace(self, x, theta, kmax, pcflags, tol):
        if self.v1.size == 0:
            self.call_conjugate_gradients(kmax, pcflags, tol)
        self.Z = self.v1
        if self.n > 1:
            self.extend_Z()
        self.first_qp = self.compute_qp(x, self.Z, theta, save_s=True)


    def call_conjugate_gradients(self, kmax, pcflags, tol):
        precon = Aprecon(self.H, self.DM, self.DG, self.n)
        precon.determine_factor(pcflags)

        pcgr = Precondition(self.H, self.DM, self.DG, self.grad)
        pcgr.solve(kmax, precon.ppvec, precon.RPCMTX, tol)

        self.v1 = pcgr.v1
        self.pcgit = pcgr.it
        self.posdef = pcgr.posdef


    def reduce_to_subspace(self, Z, grad):
        W = compute_w(self.H, self.DM, Z)
        self.MM = np.dot(Z.T, W) + np.dot(Z.T, np.dot(self.DG, Z))
        self.rhs = np.dot(Z.T, grad)

    
    def gradient_direction_evaluation(self, x, theta):
        ''' Evaluate along gradient direction '''
        ZZ = normalize(self.grad)
        self.second_qp = self.compute_qp(x, ZZ, theta)


    def compute_qp(self, x, Z, theta, save_s=False):
        self.reduce_to_subspace(Z, self.grad)
        st = self.exact_trust.compute_solution(self.rhs, self.MM)
        ss = np.dot(Z, st)
        s = abs(np.diag(self.DM)) * ss
        qp = self.QP(st, ss, s)
        self.finish_qp_evaluation(x, qp, theta)
        if save_s:
            self.set_s_values(st, ss, s)
        return qp


    def reflected_direction_evaluation(self, x, delta, theta):
        self.third_qp = self.QP(self.st, self.ss, self.s)
        self.third_qp.multiply_vars(self.mdis)
        self.keep_third_ss = self.third_qp.ss
        self.keep_thirs_s = self.third_qp.s
        
        if norm(self.third_qp.ss) < 0.9 * delta:
            nx = x + self.third_qp.s
            self.s[self.ipt] = -self.s[self.ipt]
            self.ss[self.ipt] = -self.ss[self.ipt]
            self.third_qp.update_val(self.rhs, self.MM)
            self.reflected_reduction()
            self.reflected_tau_evaluation(delta)
            self.finish_qp_evaluation(nx, self.third_qp, theta, addition=self.third_qp.val)


    def reflected_reduction(self):
            third_s = self.third_qp.s
            third_ss = self.third_qp.ss
            ng = atamult(self.H, third_s, 0)
            ng = ng + self.g 
            ngrad = np.dot(self.DM, ng) + np.dot(self.DG, third_ss)

            ZZ = normalize(self.ss)
            self.reduce_to_subspace(ZZ, ngrad)

    
    def reflected_tau_evaluation(self, delta):
        tau = self.quadratic_zero_finder(self.third_qp.ss, delta)
        self.third_qp.ss = tau * self.ss
        self.third_qp.st = tau / norm(self.third_qp.ss)
        self.third_qp.s  = abs(np.diag(self.DM)) * self.third_qp.ss


    def finish_qp_evaluation(self, x, qp, theta, addition=0):
        self.truncate_tr_solution(x, qp, theta)
        qp.multiply_vars()
        qp.update_val(self.rhs, self.MM, addition)


    def truncate_tr_solution(self, x, qp, theta):
        arg = abs(qp.s) > 0
        if np.any(np.isnan(qp.s)):
            print("Trust region step contains NaN's")
        # no truncation if s is zero length
        if not np.any(arg):
            self.mdis = 1
            qp.alpha = 1 
            return
        self.nonzero_truncation(x, qp, arg, theta)
        

    def nonzero_truncation(self, x, qp, arg, theta):
        l = self.lb
        u = self.ub
        dis = np.maximum((u[arg]-x[arg])/qp.s[arg], (l[arg]-x[arg])/qp.s[arg])
        self.mdis = min(dis)
        self.ipt = np.argmin(dis)
        mdis = self.mdis * theta
        qp.alpha = min(1, mdis)


    def extend_Z(self):
        v1 = self.v1
        if self.posdef < 1:
            v2 = np.dot(self.DM, np.sign(self.grad))
        else:
            v2 = self.grad
        v2 = normalize(v2)
        v2 = v2 - np.dot(v1, np.dot(v1.T, v2))
        if norm(v2) > np.sqrt(EPS):
            v2 = normalize(v2)
            self.Z = np.vstack([self.Z, v2]).T


    def quadratic_zero_finder(self, ss, delta):
        x = self.ss

        a = np.dot(x.T, x)
        b = 2 * np.dot(ss.T, x)
        c = np.dot(ss.T, ss) - delta**2

        numer = -(b + np.sign(b) * np.sqrt(b**2 - 4*a*c))
        r1 = numer / (2*a)
        r2 = c / (a*r1)

        tau = max(r1, r2)
        tau = min(1, tau)
        if tau <= 0:
            print("square root error in trdog/quad1d")
        return tau
    

    def choose_best_direction(self):
        qp1 = self.first_qp
        qp2 = self.second_qp
        qp3 = self.third_qp

        if qp2 <= qp1 and qp2 <= qp3:
            self.set_values(qp2)
        elif qp1 <= qp2 and qp1 <= qp3:
            self.set_values(qp1)
        else:
            self.qpval = qp3.val
            self.s = qp3.s + self.keep_thirs_s
            self.ss = qp3.ss + self.keep_third_ss

    
    def set_values(self, qp):
        self.qpval = qp.val
        self.set_s_values(qp.st, qp.ss, qp.s)


    def set_s_values(self, st, ss, s):
        self.st = st
        self.ss = ss
        self.s  = s


class Trust_Exact_Solution():
    def __init__(self, delta):
        self.delta = delta
        self.tol = 10**-12
        self.tol2 = 10**-8
        self.key = 0
        self.lambd = 0
        self.itbnd = 50
        self.count = 0


    def set_vars(self, g, H):
        self.evauluate_eigen(H)
        self.n = vector_len(g)
        self.coeff = np.zeros(self.n)
        self.laminit = -self.mineig
        self.alpha = -np.dot(self.eigvec.T, g)
        self.sig = np.sign(self.alpha[self.jmin]) + (self.alpha[self.jmin] == 0)


    def evauluate_eigen(self, H):
        try:
            self.eigval, self.eigvec = np.linalg.eig(H)
            self.mineig = min(self.eigval)
            self.jmin = np.argmin(self.eigval)
        except:
            self.eigval = np.full(1, H)
            self.eigvec = np.ones(1)
            self.mineig = self.eigval
            self.jmin = 0


    def compute_solution(self, g, H):
        self.set_vars(g, H)
        if self.mineig > 0:
            if self.positive_definite_case():
                return self.s
            self.laminit = 0
        self.indefinite_case()
        return self.s


    def positive_definite_case(self):
        self.coeff = self.alpha / self.eigval
        self.s = np.dot(self.eigvec, self.coeff)
        if norm(self.s) < 1.2 * self.delta:
            return True
        return False
    

    def indefinite_case(self):
        if self.secular_equation(self.laminit) > 0:
            # b = scipy.optimize.root_scalar(self.secular_equation, x0=self.laminit, bracket=(self.laminit, self.laminit+self.delta), max_iter=self.itbnd)
            b = self.right_find_zero()
            vval = abs(self.secular_equation(b))
            if vval <= self.tol2:
                self.state2(b)
                return
        self.state3()


    def state2(self, b, loop_flag=False):
        self.lambd = b
        s = self.compute_s(self.lambd)
        if 0.8*self.delta < norm(s) < 1.2*self.delta or loop_flag:
            self.s = s
            return s
        self.state3()
    

    def state3(self):
        self.lambd = -self.mineig
        arg = abs(self.eigval + self.lambd) < 10*EPS*np.maximum(abs(self.eigval), np.ones(self.n))
        self.alpha[arg] = 0
        s = self.compute_s(self.lambd)
        if norm(s) < 0.8 * self.delta:
            beta = np.sqrt(self.delta ** 2 - norm(s) ** 2)
            s = s + beta * self.sig * self.eigvec.T[self.jmin]
        if norm(s) > 1.2 * self.delta:
            b = self.right_find_zero()
            s = self.state2(b, loop_flag=True)
        self.s = s


    def compute_s(self, lambd):
        w = self.eigval + lambd*np.ones(self.n)
        arg1 = (w == 0) & (self.alpha == 0)
        arg2 = (w == 0) & (self.alpha != 0)
        self.coeff[w != 0] = self.alpha[w != 0] / w[w != 0]
        self.coeff[arg1] = 0
        self.coeff[arg2] = INF
        self.coeff[np.isnan(self.coeff)] = 0
        s = np.dot(self.eigvec, self.coeff)
        return s


    def secular_equation(self, lambd):
        m = vector_len(lambd)
        n = vector_len(self.eigval)
        unn = np.ones(n)
        unm = np.ones(m)

        M = self.eigval*unm + unn*lambd
        MC = M
        MM = self.alpha * unm
        M[MC != 0] = MM[MC != 0] / M[MC != 0]
        M[MC == 0] = INF * np.ones(MC[MC == 0].shape)
        M = M * M
        
        value = np.sqrt(unm / np.dot(M.T, unn))
        if value == np.NaN:
            value = 0
        value = (1 / self.delta)*unm - value
        
        return value


    def right_find_zero(self):
        # Initialization 
        self.init_right_find_zero()

        self.find_b()
        return self.b


    def init_right_find_zero(self):
        x = self.laminit
        self.itfun = 0
        if x != 0:
            dx = abs(x) / 2
        else:
            dx = 1/2

        self.init_var_right_find_zero("a", x)
        self.init_var_right_find_zero("b", x + 1)
        self.find_change_of_sign(x, dx)
        self.c = self.a
        self.fc = self.fb


    def init_var_right_find_zero(self, varname, value):
        fvalue = self.secular_equation(value)
        setattr(self, varname, value)
        setattr(self, "f" + varname, fvalue)
        self.itfun = self.itfun + 1


    def find_change_of_sign(self, x, dx):
        while (self.fa > 0) == (self.fb > 0):
            dx = 2*dx
            self.init_var_right_find_zero("b", x + dx)
            if self.itfun > self.itbnd:
                return
        return
    
    
    def find_b(self):
        while self.fb != 0:
            if (self.fb > 0) == (self.fc > 0):
                self.c = self.a
                self.fc = self.fa
                self.d = self.b - self.a
                self.e = self.d
            if abs(self.fc) < abs(self.fb):
                self.a = self.b
                self.b = self.c
                self.c = self.a
                self.fa = self.fb
                self.fb = self.fc
                self.fc = self.fa
            if self.convergence_test():
                return self.b
            self.choose_bisection_or_interpolation()
            self.find_next_point()
            

    def find_next_point(self):
        self.a = self.b
        self.fa = self.b
        if abs(self.d) > self.rf_toler:
            self.b = self.b + self.d
        else:
            if self.b > self.c:
                self.b = self.b - self.rf_toler
            else:
                self.b = self.b + self.rf_toler
        self.fb = self.secular_equation(self.b)
        self.itfun = self.itfun + 1

    
    def convergence_test(self):
        if self.itfun > self.itbnd:
            return True
        self.m = 0.5*(self.c - self.b)
        self.rf_toler = 2.0 * self.tol * max(abs(self.b), 1)
        if (abs(self.m) <= self.rf_toler) or (self.fb == 0.0):
            return True
        return False
    

    def choose_bisection_or_interpolation(self):
        if (abs(self.e) < self.rf_toler) or (abs(self.fa) <= abs(self.fb)):
            self.choose_bisection()
        self.choose_interpolation()
    

    def choose_bisection(self):
        self.d = self.m
        self.e = self.m


    def choose_interpolation(self):
        s = self.fb / self.fa
        if self.a == self.c:
            self.linear_interpolation(s)
        else:
            self.inverse_quadratic_interpolation(s)


    def linear_interpolation(self, s):
        p = 2.0 * self.m * s
        q = 1.0 - s
        self.test_interpolation(p, q)

    
    def inverse_quadratic_interpolation(self, s):
        q = self.fa / self.fc
        r = self.fb / self.fc
        p = s * (2.0*self.m*q*(q-r) - (self.b-self.a)*(r-1.0))
        q = (q - 1.0)*(r - 1.0)*(s - 1.0)
        self.test_interpolation(p, q)


    def test_interpolation(self, p, q):
        if p > 0:
            q = -q
        else:
            p = -p
        self.is_interpolated_point_acceptable(p, q)


    def is_interpolated_point_acceptable(self, p, q):
        if (2.0*p < 3.0*self.m*q - abs(self.rf_toler*q)) and p < abs(0.5*self.e*q):
            self.e = self.d
            self.d = p / q
        else:
            self.choose_bisection()
