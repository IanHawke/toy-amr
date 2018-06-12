import numpy

class box(object):
    """
    Defines the location of the box with respect to the parent.
    
    bnd: (lower, upper). Location of boundaries (integer, wrt parent)
    bbox: (lower, upper). True for physical, False for MR, boundary 
    """
    def __init__(self, bnd, bbox, Ngz):
        self.bnd = bnd
        self.bbox = bbox
        self.Npoints = 2 * (bnd[1] - bnd[0] + 1)
        self.Ngz = Ngz

class grid(object):
    def __init__(self, interval, box, parent=None):
        self.interval = interval
        self.box = box
        self.Npoints = box.Npoints
        self.Ngz = box.Ngz
        self.dx = (self.interval[1] - self.interval[0]) / self.Npoints
        self.parent = parent
    def coordinates(self):
        x_start = self.interval[0] + (0.5 - self.Ngz) * self.dx
        x_end   = self.interval[1] + (self.Ngz - 0.5) * self.dx
        return numpy.linspace(x_start, x_end, self.Npoints + 2 * self.Ngz)
    def interior_coordinates(self):
        x_start = self.interval[0] + 0.5 * self.dx
        x_end   = self.interval[1] - 0.5 * self.dx
        return numpy.linspace(x_start, x_end, self.Npoints)
        

def minmod(y):
    slope_l = numpy.zeros_like(y)
    slope_r = numpy.zeros_like(y)
    slope_l[1:] = numpy.diff(y)
    slope_r[:-1] = numpy.diff(y)
    slope = numpy.min(numpy.abs(slope_l), numpy.abs(slope_r)) * numpy.sign(slope_l)
    return slope

class patch(object):
    """
    A grid with data on it
    """
    def __init__(self, grid, model, t, Nt, parent=None):
        self.grid = grid
        self.parent = parent
        self.model = model
        self.Nvars = len(model.cons_names)
        self.Nprim = len(model.prim_names)
        self.Naux  = len(model.aux_names)
        self.prim = numpy.zeros((self.Nprim, self.grid.Npoints + 2 * self.grid.Ngz))
        self.cons = numpy.zeros((self.Nvars, self.grid.Npoints + 2 * self.grid.Ngz))
        self.aux  = numpy.zeros((self.Naux , self.grid.Npoints + 2 * self.grid.Ngz))
        self.t = t
        self.Nt = Nt
        if self.Nt:
            self.prolong_grid()
        else:
            self.prim = self.model.initial_data(self.grid.coordinates())
            self.cons, self.aux = self.model.prim2all(self.prim)
    def prolong_grid(self):
        for Nv in range(self.Nvars):
            parent_slopes = minmod(self.parent.prim[Nv, :])
            for i in range(self.grid.bnd[0], self.grid.bnd[1], 2):
                raise NotImplementedError
    