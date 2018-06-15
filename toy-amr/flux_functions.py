import numpy

def lax_friedrichs(cons_minus, cons_plus, simulation, tl):
    alpha = tl.grid.dx / tl.dt
    flux = numpy.zeros_like(cons_minus)
    prim_minus, aux_minus = simulation.model.cons2all(cons_minus, tl.prim)
    prim_plus,  aux_plus  = simulation.model.cons2all(cons_plus , tl.prim)
    f_minus = simulation.model.flux(cons_minus, prim_minus, aux_minus)
    f_plus  = simulation.model.flux(cons_plus,  prim_plus,  aux_plus )
    
    flux[:, 1:-1] = 0.5 * ( (f_plus[:,0:-2] + f_minus[:,1:-1]) + \
                    alpha * (cons_plus[:,0:-2] - cons_minus[:,1:-1]) )
    
    return flux

def upwind(cons_minus, cons_plus, simulation, patch):
    flux = numpy.zeros_like(cons_minus)
    flux[:, 1:-1] = simulation.model.riemann_problem_flux(cons_plus [:, 0:-2], 
                                                          cons_minus[:, 1:-1])
    return flux
    
