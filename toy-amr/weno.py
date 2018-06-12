import numpy

C_3 = numpy.array([1, 2]) / 3
a_3 = numpy.array([[3, -1], [1, 1]]) / 2
sigma_3 = numpy.array([[[1, 0], [-2, 1]], [[1, 0], [-2, 1]]])

C_5 = numpy.array([1, 6, 3]) / 10
a_5 = numpy.array([[11, -7, 2], [2, 5, -1], [-1, 5, 2]]) / 6
sigma_5 = numpy.array([[[40, 0, 0], 
                        [-124, 100, 0],
                        [44, -76, 16] ], 
                       [[16, 0, 0], 
                        [-52, 52, 0],
                        [20, -52, 16] ],
                       [[16, 0, 0],
                        [-76, 44, 0],
                        [100, -124, 40] ] ]) / 12

C_all = { 2 : C_3,
          3 : C_5 }
a_all = { 2 : a_3,
          3 : a_5 }
sigma_all = { 2 : sigma_3,
              3 : sigma_5 }

def weno3_upwind(q):
    order = 2
    epsilon = 1e-16
    alpha = numpy.zeros(order)
    beta = numpy.zeros(order)
    q_stencils = numpy.zeros(order)
    for k in range(order):
        for l in range(order):
            for m in range(l):
                beta[k] += sigma_3[k, l, m] * q[1 + k - l] * q[1 + k - m]     
        alpha[k] = C_3[k] / (epsilon + beta[k]**2)
        for l in range(order):
            q_stencils[k] += a_3[k, l] * q[1 + k - l] 
    w = alpha / numpy.sum(alpha)
    
    return numpy.dot(w, q_stencils)
    
def weno3(q, simulation):
    Nvars, Npoints = q.shape
    q_minus = numpy.zeros_like(q)
    q_plus  = numpy.zeros_like(q)
    for i in range(2, Npoints-2):
        for Nv in range(Nvars):
            q_plus [Nv, i] = weno3_upwind(q[Nv, i-1:i+2])
            q_minus[Nv, i] = weno3_upwind(q[Nv, i+1:i-2:-1])
    return q_minus, q_plus
            
        
def weno5_upwind(q):
    order = 3
    epsilon = 1e-16
    alpha = numpy.zeros(order)
    beta = numpy.zeros(order)
    q_stencils = numpy.zeros(order)
    for k in range(order):
        for l in range(order):
            for m in range(l):
                beta[k] += sigma_5[k, l, m] * q[2 + k - l] * q[2 + k - m]     
        alpha[k] = C_5[k] / (epsilon + beta[k]**2)
        for l in range(order):
            q_stencils[k] += a_5[k, l] * q[2 + k - l] 
    w = alpha / numpy.sum(alpha)
    
    return numpy.dot(w, q_stencils)
    
def weno5(q, simulation):
    Nvars, Npoints = q.shape
    q_minus = numpy.zeros_like(q)
    q_plus  = numpy.zeros_like(q)
    for i in range(3, Npoints-3):
        for Nv in range(Nvars):
            q_plus [Nv, i] = weno5_upwind(q[Nv, i-2:i+3])
            q_minus[Nv, i] = weno5_upwind(q[Nv, i+2:i-3:-1])
    return q_minus, q_plus
            
def weno_upwind(q, order):
    a = a_all[order]
    C = C_all[order]
    sigma = sigma_all[order]
    epsilon = 1e-16
    alpha = numpy.zeros(order)
    beta = numpy.zeros(order)
    q_stencils = numpy.zeros(order)
    for k in range(order):
        for l in range(order):
            for m in range(l+1):
                beta[k] += sigma[k, l, m] * q[order-1+k-l] * q[order-1+k-m]     
        alpha[k] = C[k] / (epsilon + beta[k]**2)
        for l in range(order):
            q_stencils[k] += a[k, l] * q[order-1+k-l] 
    w = alpha / numpy.sum(alpha)
    
    return numpy.dot(w, q_stencils)
    
def weno(q, simulation, order):
    Nvars, Npoints = q.shape
    q_minus = numpy.zeros_like(q)
    q_plus  = numpy.zeros_like(q)
    for i in range(order, Npoints-order):
        for Nv in range(Nvars):
            q_plus [Nv, i] = weno_upwind(q[Nv, i+1-order:i+order], order)
            q_minus[Nv, i] = weno_upwind(q[Nv, i+order-1:i-order:-1], order)
    return q_minus, q_plus