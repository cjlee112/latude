import numpy
import infercoop
from scipy import stats
import random

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

def pl_test(nplayers=10, theta=0.5, nrounds=20, shuffle=True, showModel=False):
    '''Basic test of PLModel:
    just generates IID players with specified theta,
    player 0 is allD
    player 1 is allC
    player 2 is TFT
    player 3 is WSLS
    Runs PL calculation on them. 
    Uses a constant (identity) reID mapping.'''
    mvals = numpy.zeros(nplayers - 1, int) # reputations
    others = stats.binom(nplayers - 2, theta)
    single = stats.binom(1, theta)
    pl = infercoop.PLModel()
    pl2 = infercoop.PLModel()
    n = myRep = wsls = 0
    old = range(nplayers - 1) # start with identity mapping
    lastroundMe = numpy.zeros(nplayers - 1)
    for i in range(nrounds):
        n += nplayers - 1 # total number of games each player has played
        mvals += others.rvs(nplayers - 1)
        lastround = single.rvs(nplayers - 1) # moves played vs. me last round
        lastround[2] = lastroundMe[2] # a TFT player
        lastround[3] = wsls = wsls * lastroundMe[3] \
            + (1 - wsls) * (1 - lastroundMe[3]) # a WSLS player
        lastroundMe = single.rvs(nplayers - 1) # moves played by me last round
        myRep += lastroundMe.sum()
        mvals += lastround # total rep of each player
        mvals[0] = lastround[0] = 0 # an allD player
        mvals[1] = n # an allC player
        lastround[1] = 1
        if shuffle:
            old, reIDs, out, outlast, outlastMe = \
                random_map(old, mvals, lastround, lastroundMe)
            #print old, reIDs
            #print mvals, out
        else: # test mapping code on trivial mapping
            reIDs = make_map(old, old)
            out = map_mvals(old, mvals)
            outlast = map_mvals(old, lastround)
            outlastMe = map_mvals(old, lastroundMe)
        pCoop = pl(reIDs, n, out, myRep,
                   outlast, outlastMe) # compute PL p(coop) for each player
        pCoop2 = pl2(range(nplayers - 1), n, mvals, myRep, 
                     lastround, lastroundMe) # compute PL p(coop) for each player
        pCoop0 = numpy.zeros(len(pCoop))
        for i,j in enumerate(old):
            pCoop0[j] = pCoop[i]
        print '\n\n------------------\nPL under shuffled vs. unshuffled model'
        print pCoop0
        print pCoop2
        if showModel:
            print '\n\nmodel data:'
            data = [d.copy() for d in pl.data]
            mapBack = numpy.zeros(len(old), int)
            for i,j in enumerate(old):
                mapBack[j] = i
            for d in data:
                infercoop.remap_reID(mapBack, d)
            print data
            print '\n\nmodel data 2:'
            print pl2.data


    
    
def make_map(old, new):
    oldIDs = numpy.zeros(len(old), int)
    for i,j in enumerate(old):
        oldIDs[j] = i
    reIDs = numpy.zeros(len(new), int)
    for i,j in enumerate(new):
        reIDs[i] = oldIDs[j]
    return reIDs

def map_mvals(new, mvals):
    out = numpy.zeros(len(mvals), int)
    for i,j in enumerate(new):
        out[i] = mvals[j]
    return out

def random_map(old, mvals, lastround, lastroundMe):
    new = [i for i in old]
    random.shuffle(new)
    reIDs = make_map(old, new)
    out = map_mvals(new, mvals)
    outlast = map_mvals(old, lastround)
    outlastMe = map_mvals(old, lastroundMe)
    return new, reIDs, out, outlast, outlastMe
