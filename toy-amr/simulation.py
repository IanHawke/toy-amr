import numpy
from matplotlib import pyplot

class simulation(object):
    def __init__(self, model, grid, rhs, timestepper, bcs, cfl=0.5):
        self.model = model
        self.Nvars = len(model.cons_names)
        self.Nprim = len(model.prim_names)
        self.Naux  = len(model.aux_names)
        self.grid = grid
        self.rhs = rhs
        self.timestepper = timestepper
        self.bcs = bcs
        self.cfl = cfl
        self.dx = grid.dx
        self.dt = cfl * self.dx # We should be dynamically measuring max(lambda)
        self.coordinates = grid.coordinates()
        self.prim0 = model.initial_data(self.coordinates)
        self.prim0 = self.prim0.reshape((self.Nprim, 
                                         self.grid.Npoints+2*self.grid.Ngz))
        self.prim = self.prim0.copy()
        self.cons0, self.aux0 = model.prim2all(self.prim0)
        self.cons = self.cons0.copy()
        self.aux  = self.aux0.copy() 
        self.t = 0
        self.fix_cons = getattr(model, "fix_cons", None)
        self.source_fprime = getattr(model, "source_fprime", None)
        self.source_guess = getattr(model, "source_guess", None)
        
    def evolve_step(self, t_end):
        alpha = self.model.max_lambda(self.cons, self.prim, self.aux)
        self.dt = self.cfl * self.dx / alpha
#        if self.timestep < 5:
#            self.dt *= 0.1
        if self.t + self.dt > t_end:
            self.dt = t_end - self.t
        self.cons = self.timestepper(self, self.cons, self.prim, self.aux)
        if self.fix_cons:
            self.cons = self.fix_cons(self.cons)
        self.cons = self.bcs(self.cons, self.grid.Npoints, self.grid.Ngz)
        self.prim, self.aux = self.model.cons2all(self.cons, self.prim)
        self.t += self.dt

    def evolve(self, t_end):
        self.timestep = 0
        while self.t < t_end:
            self.timestep += 1
            self.evolve_step(t_end)
            print('t={}'.format(self.t))

    def plot_scalar(self):
        fig = pyplot.figure()
        ax = fig.add_subplot(111)
        x = self.grid.interior_coordinates()
        ax.plot(x, self.prim[0, 
                              self.grid.Ngz:self.grid.Ngz+self.grid.Npoints])
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$q$")
        ax.set_xlim(self.grid.interval[0],self.grid.interval[1])
        qmax = numpy.max(self.prim[0,:])
        qmin = numpy.min(self.prim[0,:])
        dq = qmax - qmin
        ax.set_ylim(qmin-0.05*dq, qmax+0.05*dq)
        pyplot.title(r"Solution at $t={}$".format(self.t))
        return fig
        
    def plot_scalar_vs_initial(self):
        fig = pyplot.figure()
        ax = fig.add_subplot(111)
        x = self.grid.interior_coordinates()
        ax.plot(x, self.prim[0, 
                          self.grid.Ngz:self.grid.Ngz+self.grid.Npoints],
                'b-', label="Evolved")
        ax.plot(x, self.prim0[0, 
                          self.grid.Ngz:self.grid.Ngz+self.grid.Npoints],
                'g--', label="Initial")
        ax.legend(loc="upper left")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$q$")
        ax.set_xlim(self.grid.interval[0],self.grid.interval[1])
        qmax = numpy.max(self.prim[0,:])
        qmin = numpy.min(self.prim[0,:])
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
        
    def error_norm(self, norm):
        if norm=='inf' or norm==numpy.inf:
            return numpy.max(numpy.abs(self.prim[0,:] - self.prim0[0,:]))
        else:
            return (numpy.sum(numpy.abs(self.prim[0,:] - self.prim0[0,:])**norm)*self.dx)**(1/norm)
        