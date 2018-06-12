import numpy

class sr_swe(object):
    """
    No source
    """
    
    def __init__(self, initial_data):
        self.Nvars = 2
        self.initial_data = initial_data
        self.prim_names = (r"$\Phi$", r"$v$")
        self.cons_names = (r"$D$", r"$S$")
        self.aux_names = (r"$W$",)
        
    def prim2all(self, prim):
        phi = prim[0, :]
        v   = prim[1, :]
        W   = 1 / numpy.sqrt(1 - v**2)
        cons = numpy.zeros_like(prim)
        cons[0, :] = phi * W
        cons[1, :] = phi * W**2 * v
        aux = numpy.zeros((1, prim.shape[1]))
        aux[0, :] = W
        return cons, aux
    
    def cons2all(self, cons, prim_old):
        D = cons[0, :]
        S   = cons[1, :]
        W = numpy.sqrt((S**2 + D**2) / D**2)
        phi = D / W
        v = S / D / W
        prim = numpy.zeros_like(cons)
        prim[0, :] = phi
        prim[1, :] = v
        aux = numpy.zeros((1, prim.shape[1]))
        aux[0, :] = W
        return prim, aux
        
    def flux(self, cons, prim, aux):
        phi = prim[0, :]
        v   = prim[1, :]
        f = numpy.zeros_like(cons)
        f[0, :] = cons[0, :] * v
        f[1, :] = cons[1, :] * v + 0.5 * phi**2
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
