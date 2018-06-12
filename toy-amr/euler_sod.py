# Sod shock tube

import numpy
from models import euler_gamma_law
from bcs import outflow
from simulation import simulation
from methods import vanleer_lf
from rk import rk3
from grid import grid
from matplotlib import pyplot

Ngz = 4
Npoints = 800
L = 1
interval = grid([-L, L], Npoints, Ngz)

rhoL = 1
pL = 1
rhoR = 0.125
pR = 0.1
epsL = pL / rhoL / (5/3 - 1)
epsR = pR / rhoR / (5/3 - 1)
qL = numpy.array([rhoL, 0, epsL])
qR = numpy.array([rhoR, 0, epsR])
model = euler_gamma_law.euler_gamma_law(initial_data = euler_gamma_law.initial_riemann(qL, qR))

sim = simulation(model, interval, vanleer_lf, rk3, outflow, cfl=0.5)
sim.evolve(0.4)
sim.plot_system()
pyplot.show()
