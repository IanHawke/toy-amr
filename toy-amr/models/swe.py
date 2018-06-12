import numpy

class swe(object):
    """
    No source
    """
    
    def __init__(self, initial_data):
        self.Nvars = 2
        self.initial_data = initial_data
        self.prim_names = (r"$\Phi$", r"$u$")
        self.cons_names = (r"$\Phi$", r"$\Phi u$")
        self.aux_names = ()
        
    def prim2all(self, prim):
        phi = prim[0, :]
        u   = prim[1, :]
        cons = numpy.zeros_like(prim)
        cons[0, :] = phi
        cons[1, :] = phi * u
        aux = numpy.zeros((0, prim.shape[1]))
        return cons, aux
    
    def cons2all(self, cons, prim_old):
        prim = numpy.zeros_like(cons)
        prim[0, :] = cons[0, :]
        prim[1, :] = cons[1, :] / cons[0, :]
        aux = numpy.zeros((0, prim.shape[1]))
        return prim, aux
        
    def flux(self, cons, prim, aux):
        phi = prim[0, :]
        u   = prim[1, :]
        f = numpy.zeros_like(cons)
        f[0, :] = cons[0, :] * u
        f[1, :] = cons[1, :] * u + 0.5 * phi**2
        return f
        
    def max_lambda(self, cons, prim, aux):
        return 1.0
        
    def riemann_problem_flux(self, q_L, q_R):
        """
        Not implemented - cost
        """
        raise NotImplementedError('Exact Riemann solver is too expensive.')

def initial_riemann(ql, qr):
    return lambda x : numpy.where(x < 0.0,
                                  ql[:,numpy.newaxis]*numpy.ones((2,len(x))),
                                  qr[:,numpy.newaxis]*numpy.ones((2,len(x))))
