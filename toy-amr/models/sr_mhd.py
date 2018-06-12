import numpy
from scipy.optimize import fsolve

class sr_mhd_gamma_law(object):
    """
    No source
    """
    
    def __init__(self, initial_data, gamma = 5/3):
        self.gamma = gamma
        self.Nvars = 8
        self.Nprim = 8
        self.Naux = 11
        self.initial_data = initial_data
        self.prim_names = (r"$\rho$", r"$v_x$", r"$v_y$", r"$v_z$", 
                           r"$\epsilon$", r"$B_x$", r"$B_y$", r"$B_z$")
        self.cons_names = (r"$D$", r"$S_x$", r"$S_y$", r"$S_z$", r"$\tau$",
                           r"$B_x$", r"$B_y$", r"$B_z$")
        self.aux_names = (r"$p$", r"$W$", r"$h$", r"$c_s$", r"$B^{(4)}_0$",
                          r"$B^{(4)}_x$", r"$B^{(4)}_y$", r"$B^{(4)}_z$",
                          r"$B^{(4)}$", r"$p_{tot}$", r"$h_{tot}$")
        
    def prim2all(self, prim):
        rho = prim[0, :]
        vx  = prim[1, :]
        vy  = prim[2, :]
        vz  = prim[3, :]
        eps = prim[4, :]
        Bx  = prim[5, :]
        By  = prim[6, :]
        Bz  = prim[7, :]
        v2 = vx**2 + vy**2 + vz**2
        W = 1 / numpy.sqrt(1 - v2)
        p = (self.gamma - 1) * rho * eps
        h = 1 + eps + p / rho
        cs = numpy.sqrt(self.gamma * (self.gamma - 1) * eps / (1 + self.gamma * eps))
        B4_0 = W * (Bx * vx + By * vy + Bz * vz)
        B4_x = Bx / W + B4_0 * vx
        B4_y = By / W + B4_0 * vy
        B4_z = Bz / W + B4_0 * vz
        B4 = (Bx**2 + By**2 + Bz**2) / W**2 + (Bx * vx + By * vy + Bz * vz)**2
        p_tot = p + B4 / 2
        h_tot = h + B4 / rho
        cons = numpy.zeros_like(prim)
        cons[0, :] = rho * W
        cons[1, :] = rho * h_tot * W**2 * vx - B4_0 * B4_x
        cons[2, :] = rho * h_tot * W**2 * vy - B4_0 * B4_y
        cons[3, :] = rho * h_tot * W**2 * vz - B4_0 * B4_z
        cons[4, :] = rho * h_tot * W**2 - p_tot - rho * W - B4_0**2
        cons[5, :] = Bx
        cons[6, :] = By
        cons[7, :] = Bz
        aux = numpy.zeros((self.Naux, prim.shape[1]))
        aux[0, :] = p
        aux[1, :] = W
        aux[2, :] = h
        aux[3, :] = cs
        aux[4, :] = B4_0
        aux[5, :] = B4_x
        aux[6, :] = B4_y
        aux[7, :] = B4_z
        aux[8, :] = B4
        aux[9, :] = p_tot
        aux[10, :] = h_tot
        return cons, aux
        
    def cons_fn(self, guess, D, tau, S2, B2, SB):
        v2, omega = guess
        W = 1 / numpy.sqrt(1 - v2)
        rho = D / W
        H = omega / W**2
        p = (self.gamma - 1) / self.gamma * (H - rho)
        if rho < 0 or p < 0 or v2 >= 1:
            residual = 1e6 * numpy.ones_like(guess)
        else:
            residual = numpy.zeros_like(guess)
            residual[0] = -S2 + v2 * (B2 + omega)**2 - \
                SB**2 * (B2 + 2*omega) / omega**2
            residual[1] = (tau + D) - B2 * (1 + v2) / 2 + SB**2/(2*omega**2) - omega + p
        return residual
    
    def cons2all(self, cons, prim_old):
        Np = cons.shape[1]
        prim = numpy.zeros_like(cons)
        aux = numpy.zeros((self.Naux, Np))
        for i in range(Np):
            D   = cons[0, i]
            Sx  = cons[1, i]
            Sy  = cons[2, i]
            Sz  = cons[3, i]
            tau = cons[4, i]
            Bx  = cons[5, i]
            By  = cons[6, i]
            Bz  = cons[7, i]
            S2  = Sx**2 + Sy**2 + Sz**2
            B2  = Bx**2 + By**2 + Bz**2
            SB = Sx*Bx + Sy*By + Sz*Bz
            v2 = numpy.sum(prim_old[1:4, i]**2)
            W = 1 / numpy.sqrt(1 - v2)
            omega = prim_old[0, i] * (1 + self.gamma * prim_old[4, i]) * W**2
            initial_guess = [v2, omega]
            v2, omega = fsolve(self.cons_fn, initial_guess,
                               args = (D, tau, S2, B2, SB))
            W = 1 / numpy.sqrt(1 - v2)
            rho = D / W
            eps = (omega / W**2 - rho) / (rho * self.gamma)
            p = (self.gamma - 1) * rho * eps
            h = 1 + eps + p / rho
            cs = numpy.sqrt(self.gamma * (self.gamma - 1) * eps / (1 + self.gamma * eps))
            vx = 1 / (omega + B2) * (Sx + SB / omega * Bx)
            vy = 1 / (omega + B2) * (Sy + SB / omega * By)
            vz = 1 / (omega + B2) * (Sz + SB / omega * Bz)
            B4_0 = W * (Bx * vx + By * vy + Bz * vz)
            B4_x = Bx / W + B4_0 * vx
            B4_y = By / W + B4_0 * vy
            B4_z = Bz / W + B4_0 * vz
            B4 = (Bx**2 + By**2 + Bz**2) / W**2 + (Bx * vx + By * vy + Bz * vz)**2
            p_tot = p + B4 / 2
            h_tot = h + B4 / rho
            prim[0, i] = rho
            prim[1, i] = vx
            prim[2, i] = vy
            prim[3, i] = vz
            prim[4, i] = eps
            prim[5, i] = Bx
            prim[6, i] = By
            prim[7, i] = Bz
            aux[0, i] = p
            aux[1, i] = W
            aux[2, i] = h
            aux[3, i] = cs
            aux[4, i] = B4_0
            aux[5, i] = B4_x
            aux[6, i] = B4_y
            aux[7, i] = B4_z
            aux[8, i] = B4
            aux[9, i] = p_tot
            aux[10, i] = h_tot
        return prim, aux
        
    def flux(self, cons, prim, aux):
        tau = cons[4, :]
        vx  = prim[1, :]
        Bx  = prim[5, :]
        p_tot = aux[9, :]
        f = numpy.zeros_like(cons)
        f[0, :] = cons[0, :] * vx
        f[1:4, :] = cons[1:4, :] * vx - Bx * aux[5:8, :] / aux[1, :]
        f[1, :] += p_tot
        f[4, :] = (tau + p_tot) * vx - aux[4, :] * Bx / aux[1, :]
        f[5:8, :] = vx * cons[5:8, :] - prim[1:4, :] * Bx
        return f
        
    def fix_cons(self, cons):
        Np = cons.shape[1]
        minvals = 1e-10 * numpy.ones((1,Np))
        cons[0, :] = numpy.maximum(cons[0, :], minvals)
        return cons
        
    def max_lambda(self, cons, prim, aux):
        """
        Laziness - speed of light
        """
        return 1
        
    def riemann_problem_flux(self, q_L, q_R):
        """
        Not implemented - cost
        """
        raise NotImplementedError('Exact Riemann solver is too expensive.')

def initial_riemann(ql, qr):
    return lambda x : numpy.where(x < 0.0,
                                  ql[:,numpy.newaxis]*numpy.ones((8,len(x))),
                                  qr[:,numpy.newaxis]*numpy.ones((8,len(x))))
