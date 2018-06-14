import numpy

class box(object):
    """
    Defines the location of the box with respect to the parent.
    
    bnd: (lower, upper). Location of boundaries (integer, wrt parent, 
                             Python style [upper bound excluded])
    bbox: (lower, upper). True for physical, False for MR, boundary 
    """
    def __init__(self, bnd, bbox, Ngz):
        self.bnd = bnd
        self.bbox = bbox
        self.Npoints = 2 * (bnd[1] - bnd[0])
        self.Ngz = Ngz
        

class grid(object):
    def __init__(self, interval, box):
        self.interval = interval
        self.box = box
        self.Npoints = box.Npoints
        self.Ngz = box.Ngz
        self.dx = (self.interval[1] - self.interval[0]) / self.Npoints
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
        self.Npoints = grid.Npoints
        self.Ngz = grid.Ngz
        self.prim = numpy.zeros((self.Nprim, self.grid.Npoints + 2 * self.grid.Ngz))
        self.cons = numpy.zeros((self.Nvars, self.grid.Npoints + 2 * self.grid.Ngz))
        self.aux  = numpy.zeros((self.Naux , self.grid.Npoints + 2 * self.grid.Ngz))
        self.local_error = numpy.zeros(self.grid.Npoints + 2 * self.grid.Ngz)
        self.t = t
        self.Nt = Nt
        if self.Nt:
            assert(self.parent is not None)
            self.prolong_grid()
        else:
            self.prim = self.model.initial_data(self.grid.coordinates())
            self.cons, self.aux = self.model.prim2all(self.prim)
            
    def prolong_grid(self):
        for Nv in range(self.Nvars):
            parent_slopes = minmod(self.parent.prim[Nv, self.grid.box.bnd[0]-1:self.grid.box.bnd[1]+1])
            for p_i in range(self.grid.bnd[0], self.grid.bnd[1]):
                c_i = 2 * (p_i - self.grid.box.bnd[0])
                self.prim[Nv, c_i] = self.parent.prim[Nv, p_i] - 0.25 * parent_slopes[p_i]
                self.prim[Nv, c_i+1] = self.parent.prim[Nv, p_i] + 0.25 * parent_slopes[p_i]
            self.cons, self.aux = self.model.prim2all(self.prim)
            
    def restrict_grid(self):
        self.local_error = 0.0
        for Nv in range(self.Nvars):
            for p_i in range(self.grid.box.bnd[0], self.grid.box.bnd[1]):
                c_i = 2 * (p_i - self.grid.box.bnd[0])
                restricted_value = sum(self.prim[Nv, c_i:c_i+2]) / 2
                self.local_error[c_i:c_i+2] += abs(restricted_value - 
                                                self.parent.prim[Nv, p_i])
                self.parent.prim[Nv, p_i] = restricted_value
            self.parent.cons, self.parent.aux = self.model.prim2all(self.parent.prim)
            
    def regrid_patch(self, threshold):
        error_flag = self.local_error > threshold
        local_error = error_flag.copy()
        # Set ghosts to False to avoid boundary problems
        local_error[:self.Ngz] = False
        local_error[-self.Ngz:] = False
        # Pad the array: pad by number of ghosts, hardcoded
        for i in range(self.Ngz, self.Npoints+self.Ngz):
            local_error[i] = (error_flag[i] or 
                              numpy.any(error_flag[i-self.Ngz:i+self.Ngz]))
        # Re-set ghosts to False to avoid boundary problems
        local_error[:self.Ngz] = False
        local_error[-self.Ngz:] = False
        # Find edges of boxes to be refined
        starts = []
        ends = []
        for i in range(self.Ngz, self.Npoints+self.Ngz):
            if (not(local_error[i-1]) and local_error[i]):
                starts.append(i)
            elif (local_error[i] and not(local_error[i+1])):
                ends.append(i+1)
        # Gather into bounding boxes
        bnds = zip(starts, ends)
        # Create the boxes and grids
        grids = []
        for bnd in bnds:
            # Check boundaries
            bbox = [False, False]
            if bnd[0] == 0:
                bbox[0] = self.grid.box.bbox[0]
            elif bnd[1] == self.Npoints + self.Ngz:
                bbox[1] = self.grid.box.bbox[1]
            # Create the bbox
            b = box(bnd, bbox, self.Ngz)
            interval = (self.grid.interval[0] + (bnd[0] - self.Ngz) * self.grid.dx,
                        self.grid.interval[0] + (bnd[1] - self.Ngz) * self.grid.dx)
            g = grid(interval, b)
            grids.append(g)
        # Now create the patches
        patches = []
        for g in grids:
            patches.append(patch(g, self.model, self.t, self.Nt, self))
        return patches
        