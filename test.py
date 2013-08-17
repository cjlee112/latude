import numpy
import infercoop
from scipy import stats

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

def pl_test(nplayers=10, theta=0.5, nrounds=20):
    '''Basic test of PLModel:
    just generates IID players with specified theta,
    player 0 is allD
    player 1 is allC
    Runs PL calculation on them. 
    Uses a constant (identity) reID mapping.'''
    mvals = numpy.zeros(nplayers, int) # reputations
    others = stats.binom(nplayers - 2, theta)
    single = stats.binom(1, theta)
    pl = infercoop.PLModel()
    n = 0
    for i in range(nrounds):
        n += nplayers - 1 # total number of games each player has played
        mvals += others.rvs(nplayers)
        lastround = single.rvs(nplayers) # moves played vs. me this round
        mvals += lastround # total rep of each player, mvals[-1] is my rep
        mvals[0] = lastround[0] = 0 # an allD player
        mvals[1] = n # an allC player
        lastround[1] = 1
        pCoop = pl(range(nplayers - 1), n, mvals[:-1], mvals[-1], 
                   lastround[:-1], None) # compute PL p(coop) for each player
        print pCoop
    print '\n\nFinal model data:'
    print pl.data


    
    
