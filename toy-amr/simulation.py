import numpy
from matplotlib import pyplot
import grid

class simulation(object):
    def __init__(self, model, interval, Npoints, Ngz, rhs, timestepper, bcs, cfl=0.5,
                 threshold=1, max_levels=1):
        self.model = model
        self.Nvars = len(model.cons_names)
        self.Nprim = len(model.prim_names)
        self.Naux  = len(model.aux_names)
        self.basegrid = grid.make_basegrid(interval, Npoints, Ngz)
        self.rhs = rhs
        self.timestepper = timestepper
        self.bcs = bcs
        self.cfl = cfl
        self.threshold = threshold
        self.max_levels = max_levels
        self.coordinates = self.basegrid.coordinates()
        
        self.t = 0.0
        self.iterations = numpy.zeros(max_levels)
        self.patches = [[grid.patch(self.basegrid, model, self.t)]]
        self.iteration_steps = 2**numpy.arange(max_levels-1, -1, -1)
        self.fix_cons = getattr(model, "fix_cons", None)
        self.source_fprime = getattr(model, "source_fprime", None)
        self.source_guess = getattr(model, "source_guess", None)
        
    def evolve_level_one_step(self, dt, level):
        """
        This should be where a lot of the work happens.
        
        You evolve a given level. If there is a finer level you recurse down to
        that one, evolve that until it's at the same time as this level.
        
        Having evolved, you restrict data from finer levels to coarser. Then
        you regrid levels that are at the same time (the error measure is
        computed when restricting). By the coarse -> fine -> coarse unwinding
        of the recursion this *should* guarantee that all patches are properly
        nested.
        """
        for p in self.patches[level]:
            p.dt = dt
            # Flip timelevels
            p.swap_timelevels()
            tl = p.tl
            tl_p = p.tl_p
            tl.dt = dt
            tl_p.dt = dt
            tl.cons = self.timestepper(self, tl_p, dt)
            if self.fix_cons:
                tl.cons = self.fix_cons(tl.cons)
            if level == 0:
                tl.cons = self.bcs(tl.cons, tl.grid.Npoints, tl.grid.Ngz)
            else:
                p.prolong_boundary()
            tl.prim, tl.aux = self.model.cons2all(tl.cons, tl.prim)
            if level < len(self.patches)-1:
                self.evolve_level_one_step(dt/2, level+1)
                self.evolve_level_one_step(dt/2, level+1)
        self.iterations[level] += self.iteration_steps[level]
        if (level > 0) and (self.iterations[level-1] == self.iterations[level]):
            for p in self.patches[level]:
                p.restrict_patch()
            for p in self.patches[level]:
                p.regrid_patch()
        
    def evolve(self, t_end):
        while self.t < t_end:
            dt = self.cfl * self.basegrid.dx # Not varying timesteps
            if self.t + dt > t_end:
                dt = t_end - self.t
            self.evolve_level_one_step(dt, 0)
            self.t += dt
            print('t={}'.format(self.t))

    def plot_scalar(self):
        fig = pyplot.figure()
        ax = fig.add_subplot(111)
        qmax = -1000000
        qmin =  1000000
        for level in self.patches:
            for patch in level:
                x = patch.grid.interior_coordinates()
                ax.plot(x, patch.tl.prim[0, 
                              patch.grid.Ngz:patch.grid.Ngz+patch.grid.Npoints])
            qmax = max(qmax, numpy.max(patch.tl.prim[0,:]))
            qmin = min(qmin, numpy.min(patch.tl.prim[0,:]))
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$q$")
        ax.set_xlim(self.patches[0][0].grid.interval[0],self.patches[0][0].grid.interval[1])
        dq = qmax - qmin
        ax.set_ylim(qmin-0.05*dq, qmax+0.05*dq)
        pyplot.title(r"Solution at $t={}$".format(self.t))
        return fig
        
    # TODO: this is broken
    
    def plot_system(self):
        Nprim = len(self.model.prim_names)
        Naux  = len(self.model.aux_names)
        Nplots = Nprim + Naux
        fig = pyplot.figure(figsize=(8,3*Nplots))
        for i in range(Nprim):
            ax = fig.add_subplot(Nplots+1, 1, i+1)
            x = self.grid.interior_coordinates()
            ax.plot(x, self.prim[i, 
                              self.grid.Ngz:self.grid.Ngz+self.grid.Npoints])
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(self.model.prim_names[i])
            ax.set_xlim(self.grid.interval[0],self.grid.interval[1])
            qmax = numpy.max(self.prim[i,:])
            qmin = numpy.min(self.prim[i,:])
            dq = qmax - qmin
            ax.set_ylim(qmin-0.05*dq, qmax+0.05*dq)
        for i in range(Naux):
            ax = fig.add_subplot(Nplots+1, 1, Nprim+i+1)
            x = self.grid.interior_coordinates()
            ax.plot(x, self.aux[i, 
                              self.grid.Ngz:self.grid.Ngz+self.grid.Npoints])
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(self.model.aux_names[i])
            ax.set_xlim(self.grid.interval[0],self.grid.interval[1])
            qmax = numpy.max(self.aux[i,:])
            qmin = numpy.min(self.aux[i,:])
            dq = qmax - qmin
            ax.set_ylim(qmin-0.05*dq, qmax+0.05*dq)
        pyplot.title(r"Solution at $t={}$".format(self.t))
        pyplot.tight_layout()
        return fig
        