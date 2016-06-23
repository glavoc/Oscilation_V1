import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import derivative

class Undamped_SDOF:
    def __init__(self, x_init, v_init, k_init, m_init):
        self.x0 = x_init
        self.v0 = v_init
        self.k  = k_init
        self.m  = m_init

    def displ(self, t):
        return self.A()*np.sin(self.omega_nat()*t + self.phase())

    def velocity(self, t):
        return derivative(self.displ,t)

    def acceleration(self, t):
        return derivative(self.velocity,t)

    def omega_nat(self):
        return np.sqrt(self.k/self.m)

    def A(self):
        return np.sqrt(np.square(self.omega_nat())*np.square(self.x0) + np.square(self.v0))/self.omega_nat()
    
    def phase(self):
        if self.v0 != 0: 
            return np.arctan(self.omega_nat()*(self.x0/self.v0))
        else:
            return np.pi/2.


class damped_SDOF:
    def __init__(self, x_init, v_init, k_init, m_init, damp_ratio):
        self.x0 = x_init
        self.v0 = v_init
        self.k  = k_init
        self.m  = m_init
        self.damp_ratio = damp_ratio

        self.Real = self.x0*self.omega_damp()
        self.Imagin = self.v0 + self.damp_ratio*self.omega_nat()*self.x0

    def displ(self, t):
        return self.A()*np.power(np.e, -1.*(self.damp_ratio*self.omega_nat()*t))*np.sin(self.omega_damp()*t + self.phase())

    def velocity(self, t):
        return derivative(self.displ,t)

    def acceleration(self, t):
        return derivative(self.velocity,t)

    def omega_nat(self):
        return np.sqrt(self.k/self.m)

    def omega_damp(self):
        return self.omega_nat()*(np.sqrt(1.-np.square(self.damp_ratio)))

    def A(self):
        return np.sqrt(np.square(self.Imagin) + np.square(self.Real))/self.omega_damp()
    
    def phase(self):
        return np.arctan(self.Real/self.Imagin)
    
class Harmonic_exc():
    def __init__(self, x_init, v_init, k_init, 
                 m_init, damp_ratio, F, F_freq):
        self.x0 = x_init
        self.v0 = v_init
        self.k  = k_init
        self.m  = m_init
        self.damp_ratio = damp_ratio
        self.F = F
        self.F_freq = F_freq
        self.f0 = self.F/self.m

        self.Real = self.x0*self.omega_damp()
        self.Imagin = self.v0 + self.damp_ratio*self.omega_nat()*self.x0

    def displ(self, t):
        g = self.f0/(np.square(self.omega_nat())-np.square(self.F_freq))
        f = (self.v0/self.omega_nat())*np.sin(self.omega_nat()*t)
        s = (self.x0 - g)*np.cos(self.omega_nat()*t)
        tr =  g*np.cos(self.F_freq*t)
        return f+s+tr

    def velocity(self, t):
        return derivative(self.displ,t)

    def acceleration(self, t):
        return derivative(self.velocity,t)

    def omega_nat(self):
        return np.sqrt(self.k/self.m)

    def omega_damp(self):
        return self.omega_nat()*(np.sqrt(1.-np.square(self.damp_ratio)))

    def A(self):
        return np.sqrt(np.square(self.Imagin) + np.square(self.Real))/self.omega_damp()
    
    def phase(self):
        return np.arctan(self.Real/self.Imagin)


class Harmonic_exc_DAMP():
    def __init__(self, x_init, v_init, k_init, 
                 m_init, damp_ratio, F, F_freq):
        self.x0 = x_init
        self.v0 = v_init
        self.k  = k_init
        self.m  = m_init
        self.damp_ratio = damp_ratio
        self.F = F
        self.F_freq = F_freq
        self.f0 = self.F/self.m

        self.Real = self.x0*self.omega_damp()
        self.Imagin = self.v0 + self.damp_ratio*self.omega_nat()*self.x0

    def displ(self, t):
        f = np.power(np.e, -self.damp_ratio*self.omega_nat()*t)
        s = np.sin(self.omega_damp()*t + self.phase())
        r = np.cos(self.F_freq*t - self.theta())
        return self.A()*f*r + self.x0*r

    def velocity(self, t):
        return derivative(self.displ,t)

    def acceleration(self, t):
        return derivative(self.velocity,t)

    def X(self):
        b = np.square(np.square(self.omega_nat()) - np.square(self.F_freq))        
        c = np.square(2*self.damp_ratio*self.omega_nat()*self.F_freq)
        return self.f0/np.sqrt(b + c)

    def theta(self):
        b = np.square(self.omega_nat()) - np.square(self.F_freq)       
        c = 2*self.damp_ratio*self.omega_nat()*self.F_freq
        return np.arctan(c/b)

    def omega_nat(self):
        return np.sqrt(self.k/self.m)

    def omega_damp(self):
        return self.omega_nat()*(np.sqrt(1.-np.square(self.damp_ratio)))

    def A(self):
        return np.sqrt(np.square(self.Imagin) + np.square(self.Real))/self.omega_damp()
    
    def phase(self):
        if self.Imagin != 0:
            return np.arctan(self.Real/self.Imagin)
        else:
            return np.pi/2

def __main__():
    sdof = Harmonic_exc_DAMP(.1, .0, 30., 10., .0, 200., 0.5)
    t1 = np.arange(0.0, 500.0, .1)
    print sdof.velocity(t1)
    plt.plot(t1, sdof.displ(t1))
    plt.plot(t1, sdof.velocity(t1))
    plt.plot(t1, sdof.acceleration(t1))
    plt.show()

if __name__ == "__main__":
    __main__()
