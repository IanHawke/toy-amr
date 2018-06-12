import numpy
from scipy.optimize import brentq

class sr_euler_gamma_law(object):
    """
    No source
    """
    
    def __init__(self, initial_data, gamma = 5/3):
        self.gamma = gamma
        self.Nvars = 5
        self.Nprim = 5
        self.Naux = 4
        self.initial_data = initial_data
        self.prim_names = (r"$\rho$", r"$v_x$", r"$v_y$", r"$v_z$", r"$\epsilon$")
        self.cons_names = (r"$D$", r"$S_x$", r"$S_y$", r"$S_z$", r"$\tau$")
        self.aux_names = (r"$p$", r"$W$", r"$h$", r"$c_s$")
        
    def prim2cons(self, prim):
        rho = prim[0, :]
        vx  = prim[1, :]
        vy  = prim[1, :]
        vz  = prim[1, :]
        eps = prim[2, :]
        v2 = vx**2 + vy**2 + vz**2
        W = 1 / numpy.sqrt(1 - v2)
        p = (self.gamma - 1) * rho * eps
        h = 1 + eps + p / rho
#        cs = numpy.sqrt(self.gamma * (self.gamma - 1) * eps / (1 + self.gamma * eps))
        cons = numpy.zeros_like(prim)
        cons[0, :] = rho * W
        cons[1, :] = rho * h * W**2 * vx
        cons[2, :] = rho * h * W**2 * vy
        cons[3, :] = rho * h * W**2 * vz
        cons[4, :] = rho * h * W**2 - p - rho * W
        return cons
        
    def prim2all(self, prim):
        rho = prim[0, :]
        vx  = prim[1, :]
        vy  = prim[2, :]
        vz  = prim[3, :]
        eps = prim[4, :]
        v2 = vx**2 + vy**2 + vz**2
        W = 1 / numpy.sqrt(1 - v2)
        p = (self.gamma - 1) * rho * eps
        h = 1 + eps + p / rho
        cs = numpy.sqrt(self.gamma * (self.gamma - 1) * eps / (1 + self.gamma * eps))
        cons = numpy.zeros_like(prim)
        cons[0, :] = rho * W
        cons[1, :] = rho * h * W**2 * vx
        cons[2, :] = rho * h * W**2 * vy
        cons[3, :] = rho * h * W**2 * vz
        cons[4, :] = rho * h * W**2 - p - rho * W
        aux = numpy.zeros((self.Naux, prim.shape[1]))
        aux[0, :] = p
        aux[1, :] = W
        aux[2, :] = h
        aux[3, :] = cs
        return cons, aux
        
    def cons_fn(self, pbar, D, Sx, Sy, Sz, tau):
        if pbar > 0:
            vx = Sx / (tau + pbar + D)
            vy = Sy / (tau + pbar + D)
            vz = Sz / (tau + pbar + D)
            v2 = vx**2 + vy**2 + vz**2
            if v2 > 1 - 1e-10:
                residual = 1e6
            else:
                W = 1 / numpy.sqrt(1 - v2)
                rhoeps = (tau + D * (1 - W) + pbar * v2 / (v2 - 1)) / W**2
                residual = (self.gamma - 1) * rhoeps - pbar
        else:
            residual = 1e6
        return residual
    
    def cons2all(self, cons, prim_old):
        Np = cons.shape[1]
        prim = numpy.zeros_like(cons)
        aux = numpy.zeros((self.Naux, Np))
        for i in range(Np):
            D   = cons[0, i]
            Sx  = cons[1, i]
            Sy  = cons[2, i]
            Sz  = cons[3, i]
            tau = cons[4, i]
            S = numpy.sqrt(Sx**2 + Sy**2 + Sz**2)
            if numpy.allclose(S, 0):
                pmin = 0.1 * (self.gamma - 1) * tau
                pmax = 10.0* (self.gamma - 1) * tau
            else:
                pmin = max(S - tau - D+1e-10, 0)
                pmax = (self.gamma - 1) * tau
#            pmin = 1e-20
#            pmax = 1e20
            try:
                p = brentq(self.cons_fn, pmin, pmax,
                           args = (D, Sx, Sy, Sz, tau))
            except ValueError:
                print(D, Sx, Sy, Sz, tau)
                print(pmin, pmax)
                pvals = numpy.linspace(pmin/10, pmin*10)
                for pp in pvals:
                    print(pp, self.cons_fn(pp, D, Sx, Sy, Sz, tau))
                raise(ValueError)
            p = max(p, S - tau - D + pmin)
            vx = Sx / (tau + p + D)
            vy = Sy / (tau + p + D)
            vz = Sz / (tau + p + D)
            v2 = vx**2 + vy**2 + vz**2
            v2 = min(v2, 1-1e-10)
            W = 1 / numpy.sqrt(1 - v2)
            rho = D / W
            eps = p / (self.gamma - 1) / rho
            h = 1 + eps + p / rho
            cs = numpy.sqrt(self.gamma * (self.gamma - 1) * eps / (1 + self.gamma * eps))
            prim[0, i] = rho
            prim[1, i] = vx
            prim[2, i] = vy
            prim[3, i] = vz
            prim[4, i] = eps
            aux[0, i]  = p
            aux[1, i] = W
            aux[2, i] = h
            aux[3, i] = cs
        return prim, aux
        
    def flux(self, cons, prim, aux):
        tau = cons[4, :]
        vx  = prim[1, :]
        p   = aux[0, :]
        f = numpy.zeros_like(cons)
        f[0, :] = cons[0, :] * vx
        f[1:4, :] = cons[1:4, :] * vx
        f[1, :] += p
        f[4, :] = (tau + p) * vx
        return f
        
    def fix_cons(self, cons):
        Np = cons.shape[1]
        minvals = 1e-10 * numpy.ones((1,Np))
        cons[0, :] = numpy.maximum(cons[0, :], minvals)
        cons[4, :] = numpy.maximum(cons[4, :], minvals)
        return cons
        
    def max_lambda(self, cons, prim, aux):
        """
        Laziness - speed of light
        """
        return 1
        
    def riemann_problem_flux(self, q_L, q_R):
        """
        Not implemented - cost
        """
        raise NotImplementedError('Exact Riemann solver is too expensive.')

def initial_riemann(ql, qr):
    return lambda x : numpy.where(x < 0.0,
                                  ql[:,numpy.newaxis]*numpy.ones((5,len(x))),
                                  qr[:,numpy.newaxis]*numpy.ones((5,len(x))))
