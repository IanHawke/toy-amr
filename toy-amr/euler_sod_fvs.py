# Sod shock tube

import numpy
from models import euler_gamma_law
from bcs import outflow
from simulation import simulation
from methods import fvs_method
from rk import rk3
from grid import grid
from matplotlib import pyplot

Ngz = 3
Npoints = 200
L = 1
interval = grid([-L, L], Npoints, Ngz)

rhoL = 1
pL = 1
rhoR = 0.125
pR = 0.1
gamma = 1.4
epsL = pL / rhoL / (gamma - 1)
epsR = pR / rhoR / (gamma - 1)
qL = numpy.array([rhoL, 0, epsL])
qR = numpy.array([rhoR, 0, epsR])
model = euler_gamma_law.euler_gamma_law(initial_data = euler_gamma_law.initial_riemann(qL, qR), gamma=gamma)

sim = simulation(model, interval, fvs_method(2), rk3, outflow, cfl=0.5)
sim.evolve(0.4)
sim.plot_system()
pyplot.show()
