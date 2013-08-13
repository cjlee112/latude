import numpy

gm = numpy.array(((0, -3), (1, -2)))

gmr = numpy.array(((0, -3), (1, -2)))

gmd = dict(hh=0, hs=-3, sh=1, ss=-2)

def payout(n, k):
    return numpy.dot(gm, numpy.array((n - k, k)))

def tft_r_payout(r, n, k):
    '''tuple of payouts for (tft player, r player), assuming k r-players,
    and n-k tft-players'''
    return ((r * gmd['hh'] + (1.-r)* gmd['hs']) * k,
            (r * r * gmd['hh'] + r * (1. - r) * (gmd['hs'] + gmd['sh'])
             + (1. - r) * (1. - r) * gmd['ss']) * (k - 1) +
            (r * gmd['hh'] + (1.-r)* gmd['sh']) * (n - k))



def r1_r2_payout(r1, r2):
    return r1 * (r2 * gmd['hh'] + (1. - r2) * gmd['hs']) \
        + (1. - r1) * (r2 * gmd['sh'] + (1. - r2) * gmd['ss'])


def n_k_payout(n, k, r1, r2):
    return ((k - 1) * r1_r2_payout(r1, r1) + (n - k) * r1_r2_payout(r1, r2),
            k * r1_r2_payout(r2, r1) + (n - k - 1) * r1_r2_payout(r2, r2))

def coop_payout(n, k, r1, r2):
    return ((k - 1) * r1_r2_payout(r1, r1) + (n - k) * r1_r2_payout(r2, 0.),
            k * r1_r2_payout(0., r2) + (n - k - 1) * r1_r2_payout(0., 0.))

