import numpy

class euler_gamma_law(object):
    """
    No source
    """
    
    def __init__(self, initial_data, gamma = 5/3):
        self.gamma = gamma
        self.Nvars = 3
        self.initial_data = initial_data
        self.prim_names = (r"$\rho$", r"$v$", r"$\epsilon$")
        self.cons_names = (r"$\rho$", r"$\rho v$", r"$E$")
        self.aux_names = (r"$p$",)
        
    def prim2all(self, prim):
        rho = prim[0, :]
        v   = prim[1, :]
        eps = prim[2, :]
        cons = numpy.zeros_like(prim)
        cons[0, :] = rho
        cons[1, :] = rho * v
        cons[2, :] = rho * (0.5 * v**2 + eps)
        p = rho * eps * (self.gamma - 1)
        aux = numpy.zeros((1, prim.shape[1]))
        aux[0, :] = p
        return cons, aux
    
    def cons2all(self, cons):
        rho = cons[0, :]
        v   = cons[1, :] / rho
        eps = cons[2, :] / rho - 0.5 * v**2
        prim = numpy.zeros_like(cons)
        prim[0, :] = rho
        prim[1, :] = v
        prim[2, :] = eps
        p = rho * eps * (self.gamma - 1)
        aux = numpy.zeros((1, prim.shape[1]))
        aux[0, :] = p
        return prim, aux
        
    def flux(self, cons, prim, aux):
        E   = cons[2, :]
        rho = prim[0, :]
        v   = prim[1, :]
        p   = aux[0, :]
        f = numpy.zeros_like(cons)
        f[0, :] = rho * v
        f[1, :] = rho * v**2 + p
        f[2, :] = (E + p) * v
        return f
        
    def max_lambda(self, cons, prim, aux):
        cs = numpy.sqrt(self.gamma * aux[0, :] / prim[0, :])
        return numpy.max(numpy.abs(prim[1, :]) + cs)
        
    def riemann_problem_flux(self, q_L, q_R):
        """
        Not implemented - cost
        """
        raise NotImplementedError('Exact Riemann solver is too expensive.')

def initial_riemann(ql, qr):
    return lambda x : numpy.where(x < 0.0,
                                  ql[:,numpy.newaxis]*numpy.ones((3,len(x))),
                                  qr[:,numpy.newaxis]*numpy.ones((3,len(x))))
