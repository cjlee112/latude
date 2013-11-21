import random
from scipy import stats
from math import log
from numpy import linalg
import numpy

##################################################################
# information metric calculations

def binary_relent(omega, psi):
    return omega * log(omega / psi) \
        + (1. - omega) * log((1. - omega) / (1. - psi))

def omega_gain(omega, n, N):
    return binary_relent(omega, (n + 1.) / (N + 2.)) \
        - omega * binary_relent(omega, (n + 2.) / (N + 3.)) \
        - (1. - omega) * binary_relent(omega, (n + 1.) / (N + 3.))

def beta_gain(n, N, nsample=1000):
    rv = stats.beta(n + 1, N - n + 1)
    l = [omega_gain(omega, n, N) for omega in rv.rvs(nsample)]
    return sum(l) / float(nsample)

    
def initial_state():
    return dict(CC=[0,0], CD=[0,0], DC=[0,0], DD=[0,0])

def both_moves_gain(mymove, hismove, state):
    k = mymove + hismove
    n, N = state[k]
    return beta_gain(n, N)

def calc_info_gains(lastgame, state):
    n, N = state[lastgame]
    psi = (n + 1.) / (N + 2.) # probability he'll cooperate this round
    l = []
    for mymove in 'CD':
        gain = 0.
        for hismove, p in (('C', psi), ('D', 1. - psi)):
            gain += p * both_moves_gain(mymove, hismove, state)
        l.append((gain, mymove))
    l.sort()
    return l

##################################################################
# player classes
# each has get_move(lastgame) method that returns move based on lastgame

class InferencePlayer(object):
    def __init__(self):
        self.state = initial_state()
        self.lastgame = None
        
    def get_move(self, lastgame=None):
        if lastgame:
            if self.lastgame: # keep stats on opponent's move
                counts = self.state[self.lastgame]
                counts[1] += 1
                if lastgame[1] == 'C':
                    counts[0] += 1
            self.lastgame = lastgame
        gains = calc_info_gains(lastgame, self.state)
        return gains[-1][1] # choose with highest info gain
    
    def calc_relent(self, truedict):
        l = []
        for k,p in truedict.items():
            n, N = self.state[swap_moves(k)]
            l.append(binary_relent(p, (n + 1.) / (N + 2.)))
        return sum(l)

class MarkovPlayer(object):
    def __init__(self, pvec=None):
        if not pvec:
            pvec = [random.random() for i in range(4)]
        self.pdict = dict(CC=pvec[0], CD=pvec[1], DC=pvec[2], DD=pvec[3])
    def get_move(self, lastgame=None):
        p = self.pdict[lastgame]
        if random.random() <= p:
            return 'C'
        else:
            return 'D'

##################################################################
# basic simulations
 
def swap_moves(game):
    return game[1] + game[0]

def twoplayer_game(pvec=None, nround=100, lastgame = 'CC'):
    inferencePlayer = InferencePlayer()
    markovPlayer = MarkovPlayer(pvec)
    l = []
    for i in range(nround):
        Ip = inferencePlayer.calc_relent(markovPlayer.pdict)
        l.append(Ip)
        move1 = inferencePlayer.get_move(lastgame)
        move2 = markovPlayer.get_move(swap_moves(lastgame))
        lastgame = move1 + move2
    return l, inferencePlayer, markovPlayer

##################################################################
# score optimization

def exact_stationary(p,q):
    """Using the Press and Dyson Formula where p and q are the conditional probability vectors."""
    s = []
    c1 = [-1 + p[0]*q[0], p[1]*q[2], p[2]*q[1], p[3]*q[3]]
    c2 = [-1 + p[0], -1 + p[1], p[2], p[3]]
    c3 = [-1 + q[0], q[2], -1 + q[1], q[3]]
    
    for i in range(4):
        f = numpy.zeros(4)
        f[i] = 1
        m = numpy.matrix([c1,c2,c3,f])
        d = linalg.det(m)
        s.append(d)
    # normalize
    n = sum(s)
    s = numpy.array(s) / n
    return s

def stationary_dist(t, epsilon=1e-10):
    'compute stationary dist from transition matrix'
    diff = 1.
    while diff > epsilon:
        t = linalg.matrix_power(t, 2)
        w = t.sum(axis=1) # get row sums
        t /= w.reshape((len(w), 1)) # normalize each row
        m = numpy.mean(t, axis=0)
        diff = numpy.dot(m, t) - m
        diff = (diff * diff).sum()
    return m

def stationary_dist2(t, epsilon=.001):
    'compute stationary distribution using eigenvector method'
    w, v = linalg.eig(t.transpose())
    for i,eigenval in enumerate(w):
        s = numpy.real_if_close(v[:,i]) # handle small complex number errors
        s /= s.sum() # normalize
        if abs(eigenval - 1.) <= epsilon and (s >= 0.).sum() == len(s):
            return s # must have unit eigenvalue and all non-neg components
    raise ValueError('no stationary eigenvalue??')

def game_transition_matrix(myProbs, hisProbs0):
    'compute transition rate matrix for a strategy pair'
    # have to swap moves for other player...
    hisProbs = (hisProbs0[0], hisProbs0[2],hisProbs0[1], hisProbs0[3])
    l = []
    for i,myP in enumerate(myProbs):
        hisP = hisProbs[i]
        l.append((myP * hisP, myP * (1. - hisP), 
                  (1. - myP) * hisP, (1. - myP) * (1. - hisP)))
    return numpy.array(l)

def stationary_rates(myProbs, hisProbs):
    'compute expectation rates of all possible transitions for strategy pair'
    t = game_transition_matrix(myProbs, hisProbs)
    s = stationary_dist2(t)
    return [p * t[i] for (i,p) in enumerate(s)]

def stationary_score(myProbs, hisProbs, scores):
    'compute expectation score for my strategy vs. opponent strategy'
    rates = stationary_rates(myProbs, hisProbs)
    l = [scores * vec for vec in rates]
    return numpy.array(l).sum()

def generate_corners(epsilon=0.01):
    'generate all corners of 4D unit hypercube, with error rate epsilon'
    p = 1. - epsilon
    for move in range(16):
        myProbs = []
        for i in range(4):
            if (move >> i) & 1:
                myProbs.append(p)
            else:
                myProbs.append(epsilon)
        yield myProbs

def optimal_corner(hisProbs, scores, **kwargs):
    'find best strategy in response to a given 4D strategy'
    l = []
    for myProbs in generate_corners(**kwargs):
        l.append((stationary_score(myProbs, hisProbs, scores), myProbs))
    l.sort()
    return l

def all_vs_all(scores, **kwargs):
    'rank all possible strategies by their minimum score vs. all strategies'
    l = []
    for myProbs in generate_corners(**kwargs):
        l.append((min([(stationary_score(myProbs, hisProbs, scores), hisProbs)
                       for hisProbs in generate_corners(**kwargs)]), myProbs))
    l.sort()
    return l


def population_optimum(myFrac, hisFrac, hisProbs, scores, epsilon=0.05):
    'find optimal corner strategy at the specified population fraction'
    l = []
    for myProbs in generate_corners(epsilon):
        sAB = stationary_score(myProbs, hisProbs, scores)
        sBA = stationary_score(hisProbs, myProbs, scores)
        l.append((hisFrac * sAB - myFrac * sBA, myProbs))
    l.sort()
    return l[-1]
            
def population_diff(myFrac, myProbs, hisProbs, scores, epsilon=0.05):
    'compute relative score for strategy pair at specified population fraction'
    allC = (1. - epsilon, 1. - epsilon, 1. - epsilon, 1. - epsilon)
    sAA = stationary_score(allC, allC, scores)
    sBB = stationary_score(hisProbs, hisProbs, scores)
    sAB = stationary_score(myProbs, hisProbs, scores)
    sBA = stationary_score(hisProbs, myProbs, scores)
    return myFrac * sAA + (1. - myFrac) * sAB \
        - (1. - myFrac) * sBB - myFrac * sBA 
