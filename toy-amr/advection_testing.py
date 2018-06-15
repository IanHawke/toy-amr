# Advection test case

import numpy
from models import advection
from bcs import outflow
from simulation import simulation
from methods import vanleer_lf
from rk import rk3

Ngz = 3
Npoints = 10
L = 1
interval = [-L, L]

initial_identity = lambda x: x.reshape(1, len(x))

model = advection.advection(initial_data = initial_identity)

sim = simulation(model, interval, Npoints, Ngz, vanleer_lf, rk3, outflow, cfl=0.5)

patch = sim.patches[0][0]

before = patch.tl.prim.copy()

patch.local_error[:] = 1
ps = patch.regrid_patch(0.1)
patch_child = ps[0]
patch_child.restrict_patch()


print('difference after prolong and restrict', patch.tl.prim - before)