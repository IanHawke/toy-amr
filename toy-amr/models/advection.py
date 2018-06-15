import numpy

class advection(object):
    """
    No source
    """
    
    def __init__(self, initial_data, advection_v = 1.0):
        self.Nvars = 1
        self.advection_v = advection_v
        self.initial_data = initial_data
        self.prim_names = (r"$q$",)
        self.cons_names = (r"$q$",)
        self.aux_names = tuple()
        
    def prim2all(self, prim):
        q = prim[0, :]
        cons = numpy.zeros_like(prim)
        cons[0, :] = q
        aux = numpy.zeros((0, prim.shape[1]))
        return cons, aux
    
    def cons2all(self, cons, prim_old):
        prim = numpy.zeros_like(cons)
        prim[0, :] = cons[0, :]
        aux = numpy.zeros((0, prim.shape[1]))
        return prim, aux
        
    def flux(self, cons, prim, aux):
        q = prim[0, :]
        f = numpy.zeros_like(cons)
        f[0, :] = self.advection_v * q
        return f
        
    def max_lambda(self, cons, prim, aux):
        return abs(self.advection_v)
        
    def riemann_problem_flux(self, q_L, q_R):
        """
        Not implemented - cost
        """
        raise NotImplementedError('Exact Riemann solver is too expensive.')

def initial_riemann(ql, qr):
    return lambda x : numpy.where(x < 0.0,
                                  ql[:,numpy.newaxis]*numpy.ones((1,len(x))),
                                  qr[:,numpy.newaxis]*numpy.ones((1,len(x))))
