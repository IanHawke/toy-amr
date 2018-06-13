import numpy
from matplotlib import pyplot
import grid

class simulation(object):
    def __init__(self, model, basegrid, rhs, timestepper, bcs, cfl=0.5):
        self.model = model
        self.Nvars = len(model.cons_names)
        self.Nprim = len(model.prim_names)
        self.Naux  = len(model.aux_names)
        self.basegrid = basegrid
        self.rhs = rhs
        self.timestepper = timestepper
        self.bcs = bcs
        self.cfl = cfl
        self.coordinates = basegrid.coordinates()
        
        self.t = 0.0
        self.Nt = 0
        self.patches = [[grid.patch(basegrid, model, self.t, self.Nt)]]
        self.fix_cons = getattr(model, "fix_cons", None)
        self.source_fprime = getattr(model, "source_fprime", None)
        self.source_guess = getattr(model, "source_guess", None)
        
    def evolve_level_one_step(self, t_end, level):
        if level == 0:
            dt = self.cfl * self.basegrid.dx # Not varying timesteps
            if self.t + dt > t_end:
                dt = t_end - self.t
        for p in self.patches[level]:
            p.cons = self.timestepper(self, p, dt)
            if self.fix_cons:
                p.cons = self.fix_cons(p.cons)
            if level == 0:
                p.cons = self.bcs(p.cons, p.grid.Npoints, p.grid.Ngz)
            else:
                p.prolong_boundary()
            p.prim, p.aux = self.model.cons2all(p.cons, p.prim)
            if level < len(self.patches):
                self.evolve_level_one_step(self.t + dt/2, level+1)
                self.evolve_level_one_step(self.t + dt  , level+1)
        self.t += self.dt
        
    def evolve(self, t_end):
        self.Nt = 0
        while self.t < t_end:
            self.Nt += 1
            self.evolve_level_one_step(t_end, 0)
            print('t={}'.format(self.t))

    def plot_scalar(self):
        fig = pyplot.figure()
        ax = fig.add_subplot(111)
        qmax = -1000000
        qmin =  1000000
        for level in self.patches:
            for patch in level:
                x = patch.grid.interior_coordinates()
                ax.plot(x, patch.prim[0, 
                              patch.grid.Ngz:patch.grid.Ngz+patch.grid.Npoints])
            qmax = max(qmax, numpy.max(self.prim[0,:]))
            qmin = min(qmin, numpy.min(self.prim[0,:]))
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$q$")
        ax.set_xlim(self.patches[0][0].grid.interval[0],self.patches[0][0].grid.interval[1])
        dq = qmax - qmin
        ax.set_ylim(qmin-0.05*dq, qmax+0.05*dq)
        pyplot.title(r"Solution at $t={}$".format(self.t))
        return fig
        
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
        