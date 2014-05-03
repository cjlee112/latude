from mars import MarsStrategy

def zd_vector1(chi):
    return (1. - (2. * chi - 2.) / (4. * chi + 1.), 0.,
            (chi + 4.) / (4. * chi + 1.), 0.)

def zd_vector2(chi):
    return (1., (chi - 1.)/(3. * chi + 2.), 1., 2.*(chi - 1.)/(3. * chi + 2.))

def zdr_vector(chi, phi=0.1, B=3, C=1):
    return (1., phi * (B + chi * C), 1. - phi * (C + chi * B),
            phi * (1. - chi) * (B - C))

def ss_vector(chi, phi=0.1, B=3, C=1, kappa=None, lambd=0.):
    if kappa is None:
        kappa = B - C
    return (1. - phi * (1. - chi) * (B - C - kappa), 
            phi * (chi * B + C + (1. - chi) * kappa - lambd), 
            1. - phi * (chi * C + B - (1. - chi) * kappa + lambd),
            phi * (1. - chi) * kappa)



tft = dict(pvec=(1.,0.,1.,0.))
wsls = dict(pvec=(1.,0.,0.,1.))
alld = dict(pvec=(0.,0.,0.,0.))
allc = dict(pvec=(1.,1.,1.,1.))

# zd players for standard PD game matrix
zdx = dict(pvec=zd_vector1(2.))
zdgtft2 = dict(pvec=zd_vector2(2.))

# zd players for donation game matrix used by stewart and plotkin
zdr2 = dict(pvec=zdr_vector(0.5))
zdx2 = dict(pvec=ss_vector(0.5, kappa=0.))

mars = dict(klass=MarsStrategy)

__all__ = ['tft', 'wsls', 'alld', 'allc', 'zdx', 'zdgtft2', 'zdr2', 'zdx2']
