import numpy

class swe(object):
    """
    Maxwell's equations. Pretty useless. Uses Gaussian units so the constants
    scale out.
    """
    
    def __init__(self, initial_data, ):
        self.Nvars = 6
        self.initial_data = initial_data
        self.prim_names = (r"$E_x$", r"$E_y$", r"$E_z$", r"$B_x$", r"$B_y$", r"$B_z$")
        self.cons_names = (r"$E_x$", r"$E_y$", r"$E_z$", r"$B_x$", r"$B_y$", r"$B_z$")
        self.aux_names = ()
        
    def prim2all(self, prim):
        cons = prim.copy()
        aux = numpy.zeros((0, prim.shape[1]))
        return cons, aux
    
    def cons2all(self, cons, prim_old):
        prim = cons.copy()
        aux = numpy.zeros((0, prim.shape[1]))
        return prim, aux
        
    def flux(self, cons, prim, aux):
        E = prim[0:3, :]
        B = prim[3:, :]
        f = numpy.zeros_like(cons)
        f[1, :] =  B[2, :]
        f[2, :] = -B[1, :]
        f[4, :] = -E[2, :]
        f[5, :] =  E[1, :]
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
                                  ql[:,numpy.newaxis]*numpy.ones((6,len(x))),
                                  qr[:,numpy.newaxis]*numpy.ones((6,len(x))))
