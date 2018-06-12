import numpy

class euler_relaxation(object):
    """
    See the Pareschi paper
    """
    
    def __init__(self, initial_data, e = 0.97, sigma = 0.1, gamma = 5/3):
        self.e = e
        self.sigma = sigma
        self.gamma = gamma
        self.Nvars = 3
        self.initial_data = initial_data
        self.prim_names = (r"$\rho$", r"$v$", r"$\epsilon$")
        self.cons_names = (r"$\rho$", r"$\rho v$", r"$E$")
        self.aux_names = (r"$p$", r"$T$")
        
    def G(self, rho):
        nu = self.sigma**3 * rho * numpy.pi / 6
        nu_m = 0.64994
        return nu / (1 - (nu / nu_m)**(4.0 / 3.0 * nu_m))
        
    def A(self, rho):
        return 1 + 2 * (1 + self.e) * self.G(rho)
        
    def dA(self, rho):
        """
        Note: there is a factor 2 error in the Serna paper
        
        dA = pi(1+e)nu/3*(1+(4/3*num-1)*(nu/num)**(4/3*num))/(1-(nu/num)**(4/3*num))**2
        """
        nu = self.sigma**3 * rho * numpy.pi / 6
        nu_m = 0.64994
        term1 = 2*nu/rho*(1+self.e)
        term2 = (1+(4/3*nu_m-1)*(nu/nu_m)**(4/3*nu_m))
        term3 = (1 - (nu/nu_m)**(4/3*nu_m))**(-2)
        return term1 * term2 * term3 
        
    def prim2cons(self, prim):
        rho = prim[0, :]
        v   = prim[1, :]
        eps = prim[2, :]
        cons = numpy.zeros_like(prim)
        cons[0, :] = rho
        cons[1, :] = rho * v
        cons[2, :] = rho * (0.5 * v**2 + eps)
        return cons
        
    def prim2all(self, prim):
        rho = prim[0, :]
        v   = prim[1, :]
        eps = prim[2, :]
        cons = numpy.zeros_like(prim)
        cons[0, :] = rho
        cons[1, :] = rho * v
        cons[2, :] = rho * (0.5 * v**2 + eps)
        T = (self.gamma - 1) * eps
        p = rho * T * self.A(rho)
        aux = numpy.zeros((2, prim.shape[1]))
        aux[0, :] = p
        aux[1, :] = T
        return cons, aux
    
    def cons2all(self, cons):
        rho = cons[0, :]
        v   = cons[1, :] / rho
        eps = cons[2, :] / rho - 0.5 * v**2
        prim = numpy.zeros_like(cons)
        prim[0, :] = rho
        prim[1, :] = v
        prim[2, :] = eps
        T = (self.gamma - 1) * eps
        p = rho * T * self.A(rho)
        aux = numpy.zeros((2, prim.shape[1]))
        aux[0, :] = p
        aux[1, :] = T
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
        rho = prim[0, :]
        eps = prim[2, :]
        A = self.A(rho)
        dA = self.dA(rho)
        cs = numpy.sqrt((self.gamma-1) * eps * (A + rho*dA + (self.gamma-1)*A**2))
        return numpy.max(numpy.abs(prim[1, :]) + cs)
        
    def riemann_problem_flux(self, q_L, q_R):
        """
        Not implemented - cost
        """
        raise NotImplementedError('Exact Riemann solver is too expensive.')
        
    def source(self, g = -980):
        """
        Slow source - solve explicitly
        
        Note that the units are in cm/s!
        """
        def slow_source(cons, prim, aux):
            rho = prim[0, :]
            v = prim[1, :]
            s = numpy.zeros_like(cons)
            s[1, :] = rho * g
            #s[2, :] = rho * g * v
            return s
        return slow_source
        
    def relaxation_source(self, tau):
        """
        See Pareschi again
        """
        def fast_source(cons, prim, aux):
            rho = prim[0, :]
            T   = aux[1, :]
            s = numpy.zeros_like(cons)
            s[2, :] = -(1.0 - self.e**2) / tau * \
                    self.G(rho) * rho**2 * T**(3.0/2.0)
            return s
        return fast_source

def initial_riemann(ql, qr):
    return lambda x : numpy.where(x < 0.0,
                                  ql[:,numpy.newaxis]*numpy.ones((3,len(x))),
                                  qr[:,numpy.newaxis]*numpy.ones((3,len(x))))
