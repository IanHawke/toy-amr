def periodic(q, Npoints, Ngz):
    q[:, :Ngz] = q[:, -2*Ngz:-Ngz]
    q[:, -Ngz:] = q[:, Ngz:2*Ngz]
    return q
    
def outflow(q, Npoints, Ngz):
    for i in range(Ngz):
        q[:, i] = q[:, Ngz]
        q[:, Npoints+Ngz+i] = q[:, Npoints+Ngz-1]
    return q

def outflow_reflect_right(q, Npoints, Ngz):
    """
    This only works for hydro.
    """
    for i in range(Ngz):
        q[:, i] = q[:, Ngz]
        q[:, Npoints+Ngz+i] = q[:, Npoints+Ngz-1]
        q[1, Npoints+Ngz+i] =-q[1, Npoints+Ngz-1]
    return q