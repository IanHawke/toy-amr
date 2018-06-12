import numpy
from flux_functions import lax_friedrichs, upwind
from slope_limiting import minmod, slope_limited, constant, vanleer
from weno import weno, weno_upwind
from functools import partial

def rea_method(reconstruction, flux_solver):
    def rea_solver(cons, prim, aux, simulation):
        cons_m, cons_p = reconstruction(cons, simulation)
        flux = flux_solver(cons_m, cons_p, simulation)
        rhs = numpy.zeros_like(flux)
        rhs[:,1:-1] = 1/simulation.dx * (flux[:,1:-1] - flux[:,2:])
        return rhs
    return rea_solver
    
def rea_method_prim(reconstruction, flux_solver):
    def rea_solver(cons, prim, aux, simulation):
        prim_m, prim_p = reconstruction(prim, simulation)
        cons_m = simulation.model.prim2all(prim_m)[0]
        cons_p = simulation.model.prim2all(prim_p)[0]
        flux = flux_solver(cons_m, cons_p, simulation)
        rhs = numpy.zeros_like(flux)
        rhs[:,1:-1] = 1/simulation.dx * (flux[:,1:-1] - flux[:,2:])
        return rhs
    return rea_solver
    
def rea_method_source(reconstruction, flux_solver, source_term):
    def rea_solver(cons, prim, aux, simulation):
        cons_m, cons_p = reconstruction(cons, simulation)
        flux = flux_solver(cons_m, cons_p, simulation)
        rhs = source_term(cons, prim, aux)
        rhs[:,1:-1] += 1/simulation.dx * (flux[:,1:-1] - flux[:,2:])
        return rhs
    return rea_solver
    
def fvs_method(order, slow_source=None):
    def fvs_solver(cons, prim, aux, simulation):
        alpha = simulation.model.max_lambda(cons, prim, aux)
        flux = simulation.model.flux(cons, prim, aux)
        flux_p = (flux + alpha * cons) / 2
        flux_m = (flux - alpha * cons) / 2
        flux_p_r = numpy.zeros_like(flux_p)
        flux_m_l = numpy.zeros_like(flux_m)
        Nvars, Npoints = cons.shape
        for i in range(order, Npoints-order):
            for Nv in range(Nvars):
                flux_p_r[Nv, i] = weno_upwind(flux_p[Nv, i-order:i+order-1], order)
                flux_m_l[Nv, i] = weno_upwind(flux_m[Nv, i+order-1:i-order:-1], order)
        flux[:,1:-1] = flux_p_r[:,1:-1] + flux_m_l[:,1:-1]
        rhs = numpy.zeros_like(flux)
        rhs[:,1:-1] = 1/simulation.dx * (flux[:,1:-1] - flux[:,2:])
        if slow_source:
            rhs += slow_source(cons, prim, aux)
        return rhs
    return fvs_solver
        
    
constant_lf = rea_method(constant, lax_friedrichs)
constant_upwind = rea_method(constant, upwind)

slope_minmod = partial(slope_limited, limiter=minmod)
minmod_lf = rea_method(slope_minmod, lax_friedrichs)
minmod_upwind = rea_method(slope_minmod, upwind)
slope_vanleer = partial(slope_limited, limiter=vanleer)
vanleer_lf = rea_method(slope_vanleer, lax_friedrichs)
vanleer_upwind = rea_method(slope_vanleer, upwind)

weno3 = partial(weno, order=2)
weno3_lf = rea_method(weno3, lax_friedrichs)
weno3_upwind = rea_method(weno3, upwind)
weno5 = partial(weno, order=3)
weno5_lf = rea_method(weno5, lax_friedrichs)
weno5_upwind = rea_method(weno5, upwind)
