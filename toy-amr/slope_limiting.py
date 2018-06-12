import numpy

def constant(q, simulation):
    q_minus = q.copy()
    q_plus = q.copy()
    return q_minus, q_plus
    
def minmod(slope_upwd, slope_down):
    slope = numpy.zeros_like(slope_upwd)
    # Really should replace this with numpy functions:
    Nvar, Npoints = slope.shape
    for var in range(Nvar):
        for i in range(Npoints):
            if slope_upwd[var, i] * slope_down[var, i] < 0:
                slope[var, i] = 0 # This is default anyway
            elif abs(slope_upwd[var, i]) < abs(slope_down[var, i]):
                slope[var, i] = slope_upwd[var, i]
            else:
                slope[var, i] = slope_down[var, i]
    return slope
    
def vanleer(slope_upwd, slope_down):
    slope_down[numpy.abs(slope_down) < 1e-8] = -2*slope_upwd[numpy.abs(slope_down) < 1e-8]
    ratio = slope_upwd / slope_down
    phi = numpy.where(ratio > 0, 
                      numpy.minimum(2*ratio/(1+ratio), 2/(1+ratio)),
                      numpy.zeros_like(ratio))
    return 0.5 * (slope_upwd + slope_down) * phi
    
def slope_limited(q, simulation, limiter):
    q_minus = q.copy()
    q_plus  = q.copy()
    slope_upwd = numpy.zeros_like(q)
    slope_down = numpy.zeros_like(q)
    slope_upwd[:, :-1] = q[:, 1:] - q[:, :-1]
    slope_down[:, 1:]  = q[:, 1:] - q[:, :-1]
    slope = limiter(slope_upwd, slope_down)
    q_minus -= slope/2
    q_plus  += slope/2
    return q_minus, q_plus
    