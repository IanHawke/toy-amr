import numpy
from scipy.optimize import fsolve

def euler(simulation, cons, prim, aux):
    dt = simulation.dt
    rhs = simulation.rhs
    return cons + dt * rhs(cons, prim, aux, simulation)

def rk2(simulation, cons, prim, aux):
    dt = simulation.dt
    rhs = simulation.rhs
    cons1 = cons + dt * rhs(cons, prim, aux, simulation)
    cons1 = simulation.bcs(cons1, simulation.grid.Npoints, simulation.grid.Ngz)
    prim1, aux1 = simulation.model.cons2all(cons1, prim)
    return 0.5 * (cons + cons1 + dt * rhs(cons1, prim1, aux1, simulation))

def rk3(simulation, cons, prim, aux):
    dt = simulation.dt
    rhs = simulation.rhs
    cons1 = cons + dt * rhs(cons, prim, aux, simulation)
    cons1 = simulation.bcs(cons1, simulation.grid.Npoints, simulation.grid.Ngz)
    if simulation.fix_cons:
        cons1 = simulation.model.fix_cons(cons1)
    prim1, aux1 = simulation.model.cons2all(cons1, prim)
    cons2 = (3 * cons + cons1 + dt * rhs(cons1, prim1, aux1, simulation)) / 4
    cons2 = simulation.bcs(cons2, simulation.grid.Npoints, simulation.grid.Ngz)
    if simulation.fix_cons:
        cons2 = simulation.model.fix_cons(cons2)
    prim2, aux2 = simulation.model.cons2all(cons2, prim1)
    return (cons + 2 * cons2 + 2 * dt * rhs(cons2, prim2, aux2, simulation)) / 3

def rk_euler_split(rk_method, source):
    def timestepper(simulation, cons, prim, aux):
        consstar = rk_method(simulation, cons, prim, aux)
        primstar, auxstar = simulation.model.cons2all(consstar, prim)
        return consstar + simulation.dt * source(consstar, primstar, auxstar)
    return timestepper

def rk_backward_euler_split(rk_method, source):
    def timestepper(simulation, cons, prim, aux):
        consstar = rk_method(simulation, cons, prim, aux)
        primstar, auxstar = simulation.model.cons2all(consstar, prim)
        def residual(consguess, cons_star, prim_old):
            consguess = consguess.reshape(consguess.shape[0], 1)
            prim_old = prim_old.reshape(prim_old.shape[0], 1)
            cons_star = cons_star.reshape(cons_star.shape[0], 1)
            primguess, auxguess = simulation.model.cons2all(consguess, prim_old)
            return (consguess - cons_star - simulation.dt*source(consguess, primguess, auxguess)).ravel()
        consnext = numpy.zeros_like(cons)
        cons_initial_guess = consstar + \
                          0.5*simulation.dt*source(consstar,
                                                   primstar,
                                                   auxstar)
        for i in range(cons.shape[1]):
            consnext[:, i] = fsolve(residual, cons_initial_guess[:,i].ravel(),
                                    args=(consstar[:, i].ravel(), prim[:, i].ravel()))
        return numpy.reshape(consnext, cons.shape)
    return timestepper

def imex222(source, source_fprime=None, source_guess=None):
    gamma = 1 - 1/numpy.sqrt(2)
    def residual1(consguess, dt, cons, prim, simulation):
        consguess = consguess.reshape((cons.shape[0], 1))
        cons = cons.reshape((cons.shape[0], 1))
        prim = prim.reshape((prim.shape[0], 1))
        try:
            primguess, auxguess = simulation.model.cons2all(consguess, prim)
        except ValueError:
            res = 1e6 * numpy.ones_like(consguess)
            return res.ravel()
        res = consguess - cons - dt * gamma * source(consguess,
                                                     primguess, auxguess)
        if numpy.any(numpy.isnan(res)):
            res = 1e6 * numpy.ones_like(consguess)
        return res.ravel()
    def residual2(consguess, dt, cons, prim, k1, source1, simulation):
        consguess = consguess.reshape((cons.shape[0], 1))
        cons = cons.reshape((cons.shape[0], 1))
        prim = prim.reshape((prim.shape[0], 1))
        k1 = k1.reshape((cons.shape[0], 1))
        source1 = source1.reshape((cons.shape[0], 1))
        try:
            primguess, auxguess = simulation.model.cons2all(consguess, prim)
        except ValueError:
            res = 1e6 * numpy.ones_like(consguess)
            return res.ravel()
        res = (consguess - cons - dt * (k1 + (1 - 2*gamma)*source1 + \
            gamma*source(consguess, primguess, auxguess))).ravel()
        if numpy.any(numpy.isnan(res)):
            res = 1e6 * numpy.ones_like(consguess).ravel()
        return res
    def residual2_noflux(consguess, dt, cons, prim, source1, simulation):
        consguess = consguess.reshape((cons.shape[0], 1))
        cons = cons.reshape((cons.shape[0], 1))
        prim = prim.reshape((prim.shape[0], 1))
        source1 = source1.reshape((cons.shape[0], 1))
        try:
            primguess, auxguess = simulation.model.cons2all(consguess, prim)
        except ValueError:
            res = 1e6 * numpy.ones_like(consguess)
            return res.ravel()
        res = (consguess - cons - dt * ((1 - 2*gamma)*source1 + \
            gamma*source(consguess, primguess, auxguess))).ravel()
        if numpy.any(numpy.isnan(res)):
            res = 1e6 * numpy.ones_like(consguess).ravel()
        return res
#    def residual1_prime(consguess, dt, cons, prim, simulation):
#        consguess = consguess.reshape((cons.shape[0], 1))
#        jac = numpy.eye(cons.shape[0])
#        primguess, auxguess = simulation.model.cons2all(consguess, prim)
#        jac -= dt * gamma * source_fprime(consguess, primguess, auxguess)
#        return jac
#    def residual2_prime(consguess, dt, cons, prim, k1, source1, simulation):
#        """
#        Whilst the result is idential to residual1_prime, the argument list
#        is of course different
#        """
#        consguess = consguess.reshape((cons.shape[0], 1))
#        jac = numpy.eye(cons.shape[0])
#        primguess, auxguess = simulation.model.cons2all(consguess, prim)
#        jac -= dt * gamma * source_fprime(consguess, primguess, auxguess)
#        return jac
    residual1_prime = None
    def timestepper(simulation, cons, prim, aux):
        Np = cons.shape[1]
        dt = simulation.dt
        rhs = simulation.rhs
        consguess = cons.copy()
        if source_guess:
            primguess, auxguess = simulation.model.cons2all(consguess, prim)
            consguess = source_guess(consguess, primguess, auxguess)
        cons1 = numpy.zeros_like(cons)
        for i in range(Np):
            cons1[:,i] = fsolve(residual1, consguess[:,i],
                                fprime=residual1_prime,
                                args=(dt, cons[:,i], prim[:,i], simulation),
                                xtol = 1e-12)
        cons1 = simulation.bcs(cons1, simulation.grid.Npoints, simulation.grid.Ngz)
        prim1, aux1 = simulation.model.cons2all(cons1, prim)
        k1 = rhs(cons1, prim1, aux1, simulation)
        source1 = source(cons1, prim1, aux1)
        cons2 = numpy.zeros_like(cons)
        for i in range(Np):
            consguess_source = fsolve(residual2_noflux, cons1[:,i],
                                fprime=residual1_prime,
                                args=(dt, cons[:,i], prim1[:,i], source1[:,i], simulation),
                                xtol = 1e-12)
            consguess_flux = cons1[:,i] + dt * k1[:, i]
            consguess = 0.5 * (consguess_source + consguess_flux)
            cons2[:,i] = fsolve(residual2, consguess,
                                fprime=residual1_prime,
                                args=(dt, cons[:,i], prim1[:,i], k1[:,i], source1[:,i], simulation),
                                xtol = 1e-12)
        cons2 = simulation.bcs(cons2, simulation.grid.Npoints, simulation.grid.Ngz)
        prim2, aux2 = simulation.model.cons2all(cons2, prim1)
        k2 = rhs(cons2, prim2, aux2, simulation)
        source2 = source(cons2, prim2, aux2)
        return cons + simulation.dt * (k1 + k2 + source1 + source2) / 2
    return timestepper

def imex433(source):
    alpha = 0.24169426078821
    beta = 0.06042356519705
    eta = 0.12915286960590
    def residual1(consguess, dt, cons, prim, simulation):
        consguess = consguess.reshape((cons.shape[0], 1))
        cons = cons.reshape((cons.shape[0], 1))
        prim = prim.reshape((prim.shape[0], 1))
        try:
            primguess, auxguess = simulation.model.cons2all(consguess, prim)
        except ValueError:
            res = 1e6 * numpy.ones_like(consguess)
            return res.ravel()
        res = consguess - cons - dt * alpha * source(consguess,
                                                     primguess, auxguess)
        if numpy.any(numpy.isnan(res)):
            res = 1e6 * numpy.ones_like(consguess)
        return res.ravel()
    def residual2(consguess, dt, cons, prim, source1, simulation):
        consguess = consguess.reshape((cons.shape[0], 1))
        cons = cons.reshape((cons.shape[0], 1))
        prim = prim.reshape((prim.shape[0], 1))
        source1 = source1.reshape((source1.shape[0], 1))
        try:
            primguess, auxguess = simulation.model.cons2all(consguess, prim)
        except ValueError:
            res = 1e6 * numpy.ones_like(consguess)
            return res.ravel()
        res = consguess - cons - dt * (-alpha*source1 + alpha*source(consguess,
                                                                     primguess,
                                                                     auxguess))
        if numpy.any(numpy.isnan(res)):
            res = 1e6 * numpy.ones_like(consguess)
        return res.ravel()
    def residual3(consguess, dt, cons, prim, source2, k2, simulation):
        consguess = consguess.reshape((cons.shape[0], 1))
        cons = cons.reshape((cons.shape[0], 1))
        prim = prim.reshape((prim.shape[0], 1))
        source2 = source2.reshape((source2.shape[0], 1))
        k2 = k2.reshape((k2.shape[0], 1))
        try:
            primguess, auxguess = simulation.model.cons2all(consguess, prim)
        except ValueError:
            res = 1e6 * numpy.ones_like(consguess)
            return res.ravel()
        res = consguess - cons - dt * (k2 + (1-alpha)*source2 + alpha*source(consguess,
                                                                             primguess,
                                                                             auxguess))
        if numpy.any(numpy.isnan(res)):
            res = 1e6 * numpy.ones_like(consguess)
        return res.ravel()
    def residual4(consguess, dt, cons, prim, source1, source2, source3,
                  k2, k3, simulation):
        consguess = consguess.reshape((cons.shape[0], 1))
        cons = cons.reshape((cons.shape[0], 1))
        prim = prim.reshape((prim.shape[0], 1))
        source1 = source1.reshape((source1.shape[0], 1))
        source2 = source2.reshape((source2.shape[0], 1))
        source3 = source3.reshape((source3.shape[0], 1))
        k2 = k2.reshape((k2.shape[0], 1))
        k3 = k3.reshape((k3.shape[0], 1))
        try:
            primguess, auxguess = simulation.model.cons2all(consguess, prim)
        except ValueError:
            res = 1e6 * numpy.ones_like(consguess)
            return res.ravel()
        res = consguess - cons - \
            dt * ((k2 + k3)/4 + beta*source1 + eta*source2 + \
                  (1/2-beta-eta-alpha)*source3 + alpha*source(consguess,
                                                              primguess,
                                                              auxguess))
        if numpy.any(numpy.isnan(res)):
            res = 1e6 * numpy.ones_like(consguess)
        return res.ravel()
    residual1_prime = None
    residual2_prime = None
    residual3_prime = None
    residual4_prime = None
    def timestepper(simulation, cons, prim, aux):
        Np = cons.shape[1]
        dt = simulation.dt
        rhs = simulation.rhs
        consguess = cons.copy()

        cons1 = numpy.zeros_like(cons)
        for i in range(Np):
            cons1[:,i] = fsolve(residual1, consguess[:,i],
                                fprime=residual1_prime,
                                args=(dt, cons[:,i], prim[:,i], simulation),
                                xtol = 1e-12)
        cons1 = simulation.bcs(cons1, simulation.grid.Npoints, simulation.grid.Ngz)
        prim1, aux1 = simulation.model.cons2all(cons1, prim)
#        k1 = rhs(cons1, prim1, aux1, simulation)
        source1 = source(cons1, prim1, aux1)
        cons2 = numpy.zeros_like(cons)
        for i in range(Np):
            cons2[:,i] = fsolve(residual2, cons1[:,i],
                                fprime=residual2_prime,
                                args=(dt, cons[:,i], prim[:,i], source1[:,i],
                                      simulation),
                                xtol = 1e-12)
        cons2 = simulation.bcs(cons2, simulation.grid.Npoints, simulation.grid.Ngz)
        prim2, aux2 = simulation.model.cons2all(cons2, prim)
        k2 = rhs(cons2, prim2, aux2, simulation)
        source2 = source(cons2, prim2, aux2)
        cons3 = numpy.zeros_like(cons)
        for i in range(Np):
            cons3[:,i] = fsolve(residual3, cons2[:,i],
                                fprime=residual3_prime,
                                args=(dt, cons[:,i], prim[:,i], source2[:,i],
                                      k2[:,i],
                                      simulation),
                                xtol = 1e-12)
        cons3 = simulation.bcs(cons3, simulation.grid.Npoints, simulation.grid.Ngz)
        prim3, aux3 = simulation.model.cons2all(cons2, prim)
        k3 = rhs(cons3, prim3, aux3, simulation)
        source3 = source(cons3, prim3, aux3)
        cons4 = numpy.zeros_like(cons)
        for i in range(Np):
            cons4[:,i] = fsolve(residual4, cons3[:,i],
                                fprime=residual4_prime,
                                args=(dt, cons[:,i], prim[:,i], source1[:,i],
                                      source2[:,i], source3[:,i], k2[:,i], k3[:,i],
                                      simulation),
                                xtol = 1e-12)
        cons4 = simulation.bcs(cons4, simulation.grid.Npoints, simulation.grid.Ngz)
        prim4, aux4 = simulation.model.cons2all(cons4, prim)
        k4 = rhs(cons4, prim4, aux4, simulation)
        source4 = source(cons4, prim4, aux2)

        return cons + simulation.dt * (k2+k3+4*k4 + source2+source3+4*source4) / 6
    return timestepper
