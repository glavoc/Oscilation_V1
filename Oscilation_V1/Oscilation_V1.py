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

        self.a = self.x0*self.omega_damp()
        self.b = self.v0 + self.damp_ratio*self.omega_nat()*self.x0

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
        return np.sqrt(np.square(self.b) + np.square(self.a))/self.omega_damp()
    
    def phase(self):
        return np.arctan(self.a/self.b)
    
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

        self.a = self.x0*self.omega_damp()
        self.b = self.v0 + self.damp_ratio*self.omega_nat()*self.x0

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
        return np.sqrt(np.square(self.b) + np.square(self.a))/self.omega_damp()
    
    def phase(self):
        return np.arctan(self.a/self.b)

def __main__():
    sdof = Harmonic_exc(.0, .0, 11., 10., .408, 2., 0.5)
    t1 = np.arange(0.0, 20.0, .1)
    print sdof.velocity(t1)
    plt.plot(t1, sdof.displ(t1))
    plt.plot(t1, sdof.velocity(t1))
    plt.plot(t1, sdof.acceleration(t1))
    plt.show()

if __name__ == "__main__":
    __main__()
