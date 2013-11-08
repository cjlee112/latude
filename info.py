import random
from scipy import stats
from math import log

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
