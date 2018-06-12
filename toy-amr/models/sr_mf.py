import numpy
from scipy.optimize import brentq, newton

class sr_mf_gamma_law(object):
    """
    Multifluid case, following Amano 2016, but really using the variables from
    KD's implementation.
    """
    
    def __init__(self, initial_data, gamma = 5/3, 
                 m_e = 1.0, m_p = 1.0,
                 kappa_q = 1.0, kappa_m = 2.0, kappa_f = 1.0, 
                 eta = 0.0, kappa = 1.0):
        self.gamma = gamma
        self.eta = eta
        self.kappa = kappa
        self.kappa_q = kappa_q
        self.kappa_m = kappa_m
        self.kappa_f = kappa_f
        self.m_e = m_e
        self.m_p = m_p
        self.mass_frac_e = (self.m_e) / (self.m_e + self.m_p)
        self.mass_frac_p = (self.m_p) / (self.m_e + self.m_p)
        self.Nvars = 18
        self.Nprim = 18
        self.Naux = 21
        self.initial_data = initial_data
        self.prim_names = (r"$\rho_{e}$", r"$(v_x)_{e}$", r"$(v_y)_{e}$", r"$(v_z)_{e}$",
                           r"$\epsilon_{e}$",
                           r"$\rho_{p}$", r"$(v_x)_{p}$", r"$(v_y)_{p}$", r"$(v_z)_{p}$",
                           r"$\epsilon_{p}$",
                           r"$B_x$", r"$B_y$", r"$B_z$",
                           r"$E_x$", r"$E_y$", r"$E_z$",
                           r"$\Phi$", r"$\Psi$")
        self.cons_names = (r"$D_{e}$", r"$(S_x)_{diff}$", r"$(S_y)_{diff}$", r"$(S_z)_{diff}$", r"$\tau_{diff}$",
                           r"$D_{p}$", r"$(S_x)_{sum}$", r"$(S_y)_{sum}$", r"$(S_z)_{sum}$", r"$\tau_{sum}$",
                           r"$B_x$", r"$B_y$", r"$B_z$",
                           r"$E_x$", r"$E_y$", r"$E_z$",
                           r"$\Phi$", r"$\Psi$")
        self.aux_names = (r"$p_e$", r"$p_p$", 
                          r"$W_{e}$", r"$W_{p}$", r"$h_e$", r"$h_p$", 
                          r"$B^2$", r"$E^2$",
                          r"$\epsilon_{xjk} E^j B^k$",
                          r"$\epsilon_{yjk} E^j B^k$",
                          r"$\epsilon_{zjk} E^j B^k$",
                          r"$J^x$", r"$J^y$", r"$J^z$",
                          r"$\epsilon_{xjk} v^j B^k$",  
                          r"$\epsilon_{yjk} v^j B^k$" 
                          r"$\epsilon_{zjk} v^j B^k$",
                          r"$v_x$", r"$v_y$", r"$v_z$", r"$\rho$")

    def prim2all(self, prim):
        rho_e = prim[0, :]
        v_e  = prim[1:4, :]
        eps_e = prim[4, :]
        rho_p = prim[5, :]
        v_p  = prim[6:9, :]
        eps_p = prim[9, :]
        B  = prim[10:13, :]
        E  = prim[13:16, :]
        v2_e = numpy.sum(v_e**2, axis=0)
        v2_p = numpy.sum(v_p**2, axis=0)
        W_e = 1 / numpy.sqrt(1 - v2_e)
        W_p = 1 / numpy.sqrt(1 - v2_p)
        p_e = (self.gamma - 1) * rho_e * eps_e
        p_p = (self.gamma - 1) * rho_p * eps_p
        h_e = 1 + eps_e + p_e / rho_e
        h_p = 1 + eps_p + p_p / rho_p
        B2 = numpy.sum(B**2, axis=0)
        E2 = numpy.sum(E**2, axis=0)
        EcrossB = numpy.cross(E, B, axis=0)
#        EcrossB[0,:] = E[1,:] * B[2,:] - E[2,:] * B[1,:]
#        EcrossB[1,:] = E[2,:] * B[0,:] - E[0,:] * B[2,:]
#        EcrossB[2,:] = E[0,:] * B[1,:] - E[1,:] * B[0,:]
        cons = numpy.zeros_like(prim)
        cons[0, :] = rho_e * W_e
        cons[1:4, :] = rho_p * h_p * W_p**2 * v_p - \
                       rho_e * h_e * W_e**2 * v_e
        cons[4, :] = (rho_p * h_p * W_p**2 - p_p - rho_p * W_p) - \
                     (rho_e * h_e * W_e**2 - p_e - rho_e * W_e)
        cons[5, :] = rho_p * W_p
        cons[6:9, :] = self.mass_frac_p * rho_p * h_p * W_p**2 * v_p + \
                       self.mass_frac_e * rho_e * h_e * W_e**2 * v_e + \
                       self.kappa_q / self.kappa_m * EcrossB
        cons[9, :] = self.mass_frac_p * (rho_p * h_p * W_p**2 - p_p - rho_p * W_p) + \
                     self.mass_frac_e * (rho_e * h_e * W_e**2 - p_e - rho_e * W_e) + \
                     self.kappa_q / self.kappa_m * (E2 + B2) / 2
        cons[10:13, :] = B
        cons[13:16, :] = E
        cons[16:18, :] = prim[16:18, :]
        aux = numpy.zeros((self.Naux, prim.shape[1]))
        aux[0, :] = p_e
        aux[1, :] = p_p
        aux[2, :] = W_e
        aux[3, :] = W_p
        aux[4, :] = h_e
        aux[5, :] = h_p
        aux[6, :] = B2
        aux[7, :] = E2
        aux[8:11, :] = EcrossB
        J = ( rho_p * W_p * v_p - \
              rho_e * W_e * v_e) / self.kappa_q
        rho = rho_p * W_p / self.mass_frac_p + rho_e * W_e / self.mass_frac_e
        v = rho_p * W_p * v_p / self.mass_frac_p + \
            rho_e * W_e * v_e / self.mass_frac_e
        vcrossB = numpy.cross(v, B, axis=0)
#        vcrossB[0,:] = v[1,:] * B[2,:] - v[2,:] * B[1,:]
#        vcrossB[1,:] = v[2,:] * B[0,:] - v[0,:] * B[2,:]
#        vcrossB[2,:] = v[0,:] * B[1,:] - v[1,:] * B[0,:]
        aux[11:14, :] = J
        aux[14:17, :] = vcrossB
        aux[17:20, :] = v
        aux[20, :] = rho
        return cons, aux
        
    def cons_fn(self, guess, D, tau, S2):
        v2 = S2 / guess**2
        W = 1 / numpy.sqrt(1 - v2)
        rho = D / W
        p = guess - tau - D
        if p < 0 or rho < 0 or guess < 0 or v2 >= 1:
            residual = 1e6
        else:
#            residual = (1 - (self.gamma - 1) / (self.gamma * W**2)) * guess + \
#                ((self.gamma - 1) / (self.gamma * W) - 1) * D - tau
            residual = guess - (rho + self.gamma / (self.gamma - 1) * p) * W**2
        return residual
    
    def cons2all(self, cons, prim_old):
        Np = cons.shape[1]
        prim = numpy.zeros_like(cons)
        aux = numpy.zeros((self.Naux, Np))
        for i in range(Np):
            B = cons[10:13, i]
            E = cons[13:16, i]
            EcrossB = numpy.cross(E, B)
#            EcrossB[0] = E[1,i] * B[2,i] - E[2,i] * B[1,i]
#            EcrossB[1] = E[2,i] * B[0,i] - E[0,i] * B[2,i]
#            EcrossB[2] = E[0,i] * B[1,i] - E[1,i] * B[0,i]
            B2 = numpy.sum(B**2)
            E2 = numpy.sum(E**2)
            D_e = cons[0, i]
            Sdiff = cons[1:4, i]
            taudiff = cons[4, i]
            D_p = cons[5, i]
            Ssum = cons[6:9, i] - self.kappa_q / self.kappa_m * EcrossB
            tausum = cons[9, i] - self.kappa_q / self.kappa_m * (E2 + B2) / 2

            S_p = Ssum + self.mass_frac_e * Sdiff
            tau_p = tausum + self.mass_frac_e * taudiff
            S_e = S_p - Sdiff
            tau_e = tau_p - taudiff

            S2_e = numpy.sum(S_e**2)
            v2_e = numpy.sum(prim_old[1:4, i]**2)
            W_e = 1 / numpy.sqrt(1 - v2_e)
#            omega_guess = prim_old[0, i] * (1 + self.gamma * prim_old[4, i]) * W_e**2
            omega = brentq(self.cons_fn, tau_e, 10*(tau_e + D_e + prim_old[4, i]),
                           args = (D_e, tau_e, S2_e))
#            omega = newton(self.cons_fn, omega_guess,
#                           args = (D_e, tau_e, S2_e))
            v2_e = S2_e / omega**2
            W_e = 1 / numpy.sqrt(1 - v2_e)
            rho_e = D_e / W_e
            eps_e = (omega / W_e**2 - rho_e) / (rho_e * self.gamma)
            p_e = (self.gamma - 1) * rho_e * eps_e
            h_e = 1 + eps_e + p_e / rho_e
            v_e = S_e / (rho_e * h_e * W_e**2)
            
            S2_p = numpy.sum(S_p**2)
            v2_p = numpy.sum(prim_old[6:9, i]**2)
            W_p = 1 / numpy.sqrt(1 - v2_p)
#            omega_guess = prim_old[5, i] * (1 + self.gamma * prim_old[9, i]) * W_p**2
            omega = brentq(self.cons_fn, tau_p, 10*(tau_p + D_p + prim_old[9, i]),
                           args = (D_p, tau_p, S2_p))
#            omega = newton(self.cons_fn, omega_guess,
#                           args = (D_p, tau_p, S2_p))
            v2_p = S2_p / omega**2
            W_p = 1 / numpy.sqrt(1 - v2_p)
            rho_p = D_p / W_p
            eps_p = (omega / W_p**2 - rho_p) / (rho_p * self.gamma)
            p_p = (self.gamma - 1) * rho_p * eps_p
            h_p = 1 + eps_p + p_p / rho_p
            v_p = S_p / (rho_p * h_p * W_p**2)
            
            prim[0, i] = rho_e
            prim[1:4, i] = v_e
            prim[4, i] = eps_e
            prim[5, i] = rho_p
            prim[6:9, i] = v_p
            prim[9, i] = eps_p
            prim[10:13, i] = B
            prim[13:16, i] = E
            prim[16:18, i] = cons[16:18, i]
            aux[0, i] = p_e
            aux[1, i] = p_p
            aux[2, i] = W_e
            aux[3, i] = W_p
            aux[4, i] = h_e
            aux[5, i] = h_p
            aux[6, i] = B2
            aux[7, i] = E2
            aux[8:11, i] = EcrossB
            J = ( rho_p * W_p * v_p - \
                  rho_e * W_e * v_e ) / self.kappa_q
            rho = rho_p * W_p / self.mass_frac_p + rho_e * W_e / self.mass_frac_e
            v = rho_p * W_p * v_p / self.mass_frac_p + \
                rho_e * W_e * v_e / self.mass_frac_e
            vcrossB = numpy.cross(v, B)
#            vcrossB[0] = v[1,i] * B[2,i] - v[2,i] * B[1,i]
#            vcrossB[1] = v[2,i] * B[0,i] - v[0,i] * B[2,i]
#            vcrossB[2] = v[0,i] * B[1,i] - v[1,i] * B[0,i]
            aux[11:14, i] = J
            aux[14:17, i] = vcrossB
            aux[17:20, i] = v
            aux[20, i] = rho
        return prim, aux
        
    def flux(self, cons, prim, aux):
        B = cons[10:13, :]
        E = cons[13:16, :]
        rho_e = prim[0, :]
        v_e   = prim[1:4, :]
        rho_p = prim[5, :]
        v_p   = prim[6:9, :]
        p_e = aux[0, :]
        p_p = aux[1, :]
        W_e = aux[2, :]
        W_p = aux[3, :]
        h_e = aux[4, :]
        h_p = aux[5, :]
        B2  = aux[6, :]
        E2  = aux[7, :]
        EcrossB_x = aux[8, :]
#        J = aux[11:14, :]
        
        f = numpy.zeros_like(cons)
        fSp = rho_p * h_p * W_p**2 * v_p * v_p[0, :]
        fSp[0, :] += p_p
        fSe = rho_e * h_e * W_e**2 * v_e * v_e[0, :]
        fSe[0, :] += p_e
        ftaup = (rho_p * h_p * W_p**2 - rho_p * W_p) * v_p[0, :]
        ftaue = (rho_e * h_e * W_e**2 - rho_e * W_e) * v_e[0, :]
        f[0, :] = rho_e * W_e * v_e[0, :]
        f[1:4, :] = fSp - fSe
        f[4, :] = ftaup - ftaue
        f[5, :] = rho_p * W_p * v_p[0, :]
        f[6:9, :] = self.mass_frac_p * fSp + \
                    self.mass_frac_e * fSe - \
                    self.kappa_q / self.kappa_m * (E * E[0, :] + B * B[0, :])
        f[6, :] += self.kappa_q / self.kappa_m * (E2 + B2) / 2
        f[9, :] = self.mass_frac_p * ftaup + \
                  self.mass_frac_e * ftaue + \
                  self.kappa_q / self.kappa_m * EcrossB_x
        f[10, :] = cons[16, :] # Bx
        f[11, :] = -E[2, :]
        f[12, :] =  E[1, :]
        f[13, :] = cons[17, :] #Ex
        f[14, :] =  B[2, :]
        f[15, :] = -B[1, :]
        # Lagrange multipliers
        f[16, :] = B[0, :]
        f[17, :] = E[0, :]
        return f
        
#    def fix_cons(self, cons):
#        Np = cons.shape[1]
#        minvals = 1e-10 * numpy.ones((1,Np))
#        cons[0, :] = numpy.maximum(cons[0, :], minvals)
#        return cons
        
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
        
#    def source(self):
#        def slow_source(cons, prim, aux):
#            s = numpy.zeros_like(cons)
#            s[12, :] = cons[11, :] - self.kappa * cons[12, :]
#            return s
#        return slow_source
        
    def relaxation_source(self):
        """
        Simple isotropic case
        """
        def fast_source(cons, prim, aux):
            s = numpy.zeros_like(cons)
#            return s
            E = cons[13:16, :]
            rho_e = prim[0, :]
            v_e   = prim[1:4, :]
            rho_p = prim[5, :]
            v_p   = prim[6:9, :]
            W_e = aux[2, :]
            W_p = aux[3, :]
            J = aux[11:14, :]
            vcrossB = aux[14:17, :]
            v = aux[17:20, :]
            rho = aux[20, :]
#            s[6:9, :] = (rho * E + vcrossB) / self.kappa_m + \
#                        rho_p * rho_e / self.kappa_f * \
#                        (1 / self.mass_frac_e + 1 / self.mass_frac_p) * \
#                        (W_e * v_e - W_p * v_p)
            s[1:4, :] = (rho * E + vcrossB) / self.kappa_m + \
                        rho_p * rho_e / self.kappa_f * \
                        (1 / self.mass_frac_e + 1 / self.mass_frac_p) * \
                        (W_e * v_e - W_p * v_p)
            s[4, :] = 1.0 / self.kappa_m * numpy.sum(v*E) + \
                        rho_p * rho_e / self.kappa_f * \
                        (1. / self.mass_frac_e + 1. / self.mass_frac_p) * \
                        (W_e  - W_p)
            s[13:16, :] = -J
            # Extended Lagrange multipliers
            s[16, :] = - self.kappa * cons[16, :]
            s[17, :] = (cons[5,:] - cons[0,:]) / self.kappa_q - \
                       self.kappa * cons[17, :]
            
            return s
        return fast_source
#        
#    def relaxation_guess(self):
#        def guess_function(cons, prim, aux):
#            guess = cons.copy()
#            print('guess', guess.shape)
#            if self.sigma > 1:
#                mhd_result = guess.copy()
#                v = prim[1:4,:]
#                B = prim[5:8,:]
#                E = - numpy.cross(v, B)
#                mhd_result[8:11,:] = E
#                guess = guess / self.sigma + mhd_result * (1 - 1 / self.sigma)
#            return guess
#        return guess_function
    
def initial_riemann(ql, qr):
    return lambda x : numpy.where(x < 0.0,
                                  ql[:,numpy.newaxis]*numpy.ones((18,len(x))),
                                  qr[:,numpy.newaxis]*numpy.ones((18,len(x))))

def initial_random(amplitude):
    def initial_data(x):
        nx = len(x)
        init_data = numpy.random.rand(18, nx)
        init_data[0, :] *= amplitude
        init_data[1:4, :] -= 0.5
        init_data[5, :] *= amplitude
        init_data[6:9, :] -= 0.5
        init_data[10:16, :] -= 0.5
        init_data[16:, :] = 0.0
        return init_data
    return initial_data

def initial_alfven(gamma = 4.0 / 3.0, 
                   Kappa_q = 1.0, Kappa_m = 0.05066059182116889, Kappa_f = 0.0,
                   kwave = 2.0 * numpy.pi, rhobval = 1.0, pressbval = 1.0/4.0):
    def initial_data(x):
        epsbval = pressbval / (gamma-1.0) / rhobval
        hbval   = 1.0 + epsbval + pressbval / rhobval
    
        Omega2p = 1.0 / Kappa_q / Kappa_m * 4.0 * rhobval / hbval
        kbwave = numpy.sqrt(Omega2p)
        Omega = numpy.sqrt(Omega2p * (1.0 + (kwave/kbwave)**2))

        Bperp = 1.0
        Bparallel = 0.0
    
        A_vel = 2.0 / Kappa_m  / hbval / kwave
        
        new_time = 0.0
    
        rho_p = rhobval
        press_p = pressbval
        eps_p = press_p/(gamma-1.0)/rho_p
        
        rho_e = rhobval 
        press_e = press_p 
        eps_e = press_e/(gamma-1.0)/rho_e 
        
        Bvecx = Bparallel * x
        Bvecy = +Bperp*numpy.cos(-Omega*new_time + kwave * x)
        Bvecz = -Bperp*numpy.sin(-Omega*new_time + kwave * x)
        
        Evecx = 0.0
        Evecy = (Omega/kwave)*Bvecz
        Evecz = -(Omega/kwave)*(Bvecy)
        
        velxp = 0.0
        velyp = -A_vel * Bvecy/numpy.sqrt(A_vel**2 * Bperp**2 + 1.0)
    
        velzp = -A_vel * Bvecz/numpy.sqrt(A_vel**2 * Bperp**2 + 1.0)
    
        velxe = 0.0
        velye = - velyp
        velze = - velzp
    
        init_data = numpy.zeros(shape=(18,len(x)))
        init_data[0, :] += rho_e
        init_data[1, :] += velxe
        init_data[2, :] += velye
        init_data[3, :] += velze
        init_data[4, :] += eps_e    
        init_data[5, :] += rho_p
        init_data[6, :] += velxp
        init_data[7, :] += velyp
        init_data[8, :] += velzp
        init_data[9, :] += eps_p
        init_data[10,:] += Bvecx
        init_data[11,:] += Bvecy
        init_data[12,:] += Bvecz
        init_data[13,:] += Evecx
        init_data[14,:] += Evecy
        init_data[15,:] += Evecz
        return init_data
    return initial_data

