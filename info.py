import random
from scipy import stats, optimize
from math import log, exp
from numpy import linalg
import numpy
import functools
import hashlib

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

def calc_info_gains(last_move, state):
    n, N = state[last_move]
    psi = (n + 1.) / (N + 2.) # probability he'll cooperate this round
    l = []
    for mymove in 'CD':
        gain = 0.
        for hismove, p in (('C', psi), ('D', 1. - psi)):
            gain += p * both_moves_gain(mymove, hismove, state)
        l.append((gain, mymove))
    l.sort()
    return l

def calc_best_long_term_response(q, scores=[3,0,5,1]):
    f = functools.partial(stationary_score, scores=scores, hisProbs=q)
    p, _, _ = optimize.fmin_tnc(lambda p: -1* f(p), [0.5,0.5,0.5,0.5], bounds=[(0,1)]*4, approx_grad=True, messages=0, maxfun=1000)
    print q,p
    return p

##################################################################
# player classes
# each has next_move(last_move) method that returns move based on last_move

def save_counts(move, counts):
    counts[1] += 1
    if move == 'C':
        counts[0] += 1

class InferencePlayer(object):
    def __init__(self):
        self.state = initial_state()
        self.last_move = None
        self.is_first_move = True

    def first_move(self):
        return 'C'

    def _save_last_move(self, last_move):
        if last_move:
            if self.last_move: # keep stats on opponent's move
                save_counts(last_move[1], self.state[self.last_move])
            self.last_move = last_move

    def next_move(self, last_move=None):
        if self.is_first_move:
            self.is_first_move = False
            return self.first_move()
        self._save_last_move(last_move)
        gains = calc_info_gains(last_move, self.state)
        return gains[-1][1] # choose with highest info gain
    
    def calc_relent(self, truedict):
        l = []
        for k,p in truedict.items():
            n, N = self.state[swap_moves(k)]
            l.append(binary_relent(p, (n + 1.) / (N + 2.)))
        return sum(l)

enum = dict(zip(['CC','CD','DC','DD'], range(4)))

def expectation_count(myMove, lastMove, state):
    nC, n = state[lastMove]
    pC = (nC + 1.) / (n + 2.)
    return sum([p * state[myMove + hisMove][1] 
                for (hisMove, p) in (('C', pC), ('D', 1. - pC))])
        
def tie_breaker(history):
    i = ord(hashlib.md5(''.join(history)).digest()[0]) & 1 # LSB of hash
    return 'CD'[i]

def simple_infogain_move(lastMove, state, history):
    l = [(expectation_count(myMove, lastMove, state), myMove)
         for myMove in 'CD']
    l.sort()
    if l[0][0] < l[1][0]: # we have a winner!
        return l[0][1]
    else: # tie
        return tie_breaker(history)

def opponent_infogain_move(lastMove, state, history):
    lastMove = swap_moves(lastMove)
    d = {}
    for k,v in state.items():
        d[swap_moves(k)] = v
    history = [swap_moves(g) for g in history]
    return simple_infogain_move(lastMove, d, history)

def move_likelihood(pC, move):
    if move == 'C':
        return pC
    else:
        return 1. - pC

def beta_posterior(counts, move):
    return move_likelihood((counts[0] + 1.) / (counts[1] + 2.), move)

def stochastic_move(pC):
    if random.random() < pC:
        return 'C'
    else:
        return 'D'

class InfogainStrategy(object):
    _firstMove = 'C'
    def __init__(self):
        self.state = initial_state()
        self.last_move = None
        self.hisState = initial_state()
        self.history = []
        self.myIpMoves = []
        self.hisIpMoves = []
        self.mismatches = 0
    def next_move(self, epsilon=0.05, **kwargs):
        if self.last_move:
            myMove = simple_infogain_move(self.last_move, self.state, 
                                          self.history)
            hisMove = opponent_infogain_move(self.last_move, self.state, 
                                             self.history)
        else:
            myMove = hisMove = self._firstMove
        self.myIpMoves.append(myMove)
        self.hisIpMoves.append(hisMove)
        d = dict(C=1. - epsilon, D=epsilon)
        self.myIpPC = d[myMove]
        self.hisIpPC = d[hisMove]
        return myMove
    def save_outcome(self, last_move, epsilon=0.05, state=None, hisState=None):
        'get likelihood odds infogain model / binomial model for him & me'
        if state is None: # during infogain phase, just use our mutual history
            state = self.state
        if hisState is None: # he's in infogain phase, use our mutual history
            hisState = self.hisState
        if self.last_move:
            ctx = self.last_move
            hisOdds = 1. / beta_posterior(state[ctx], last_move[1])
            save_counts(last_move[1], self.state[ctx])
            ctx = swap_moves(ctx)
            myOdds = 1. / beta_posterior(hisState[ctx], last_move[0])
            save_counts(last_move[0], self.hisState[ctx])
        else: # no history, so use uninformative prior
            hisOdds = myOdds = 2.
        myOdds *= move_likelihood(self.myIpPC, last_move[0])
        hisOdds *= move_likelihood(self.hisIpPC, last_move[1])
        self.last_move = last_move
        self.history.append(last_move)
        if self.hisIpMoves and self.hisIpMoves[-1] != last_move[1]:
            self.mismatches += 1
        return (hisOdds, myOdds)
    def match_pval(self, epsilon=0.05):
        'get p-value that opponent is an inference player (plays like me)'
        if self.mismatches == 0:
            return 1.
        b = stats.binom(len(self.hisIpMoves), epsilon)
        return b.sf(self.mismatches - 1)

class GroupMaxStrategy(InfogainStrategy):
    def start(self, myIpLOD=0., tau=0.01):
        self.pPartner = 1. / (exp(-myIpLOD) + 1.)
        self.tau = tau
        self.last_move2 = self.last_move # keep history for 2 moves...
    def next_move(self, epsilon=0.05, hisP=0, myP=0, optimalStrategy=None):
        if hisP > 0.5: # cooperate with any inference player
            myMove = 'C'
            self.myIpPC = 1. - epsilon
        else: # apply optimal strategy vs. group
            self.myIpPC = add_noise(optimalStrategy[self.last_move], epsilon)
            myMove = stochastic_move(optimalStrategy[self.last_move])
        # 2-state HMM for whether inf-player is partnering with us or not
        p1 = ((1. - self.tau) * self.pPartner 
              + self.tau * (1. - self.pPartner)) \
            * move_likelihood(1. - epsilon, self.last_move[1]) # partner
        ctx = swap_moves(self.last_move2)
        p2 = (self.tau * self.pPartner 
              + (1. - self.tau) * (1. - self.pPartner)) \
              * move_likelihood(add_noise(optimalStrategy[ctx], epsilon),
                                self.last_move[1]) # not partnering with me
        self.pPartner = p = p1 / (p1 + p2)
        pC = p + (1. - p) * optimalStrategy[swap_moves(self.last_move)]
        self.hisIpPC = add_noise(pC, epsilon) # p(inf-player cooperates w/ me)
        self.last_move2 = self.last_move
        return myMove

class MarkovStrategy(object):
    def __init__(self, pvec=None, firstMove='C'):
        if not pvec:
            pvec = [random.random() for i in range(4)]
        self.pdict = dict(CC=pvec[0], CD=pvec[1], DC=pvec[2], DD=pvec[3])
        self._firstMove = firstMove
        self.last_move = None

    def next_move(self, epsilon=0.05):
        if not self.last_move:
            return self._firstMove
        return stochastic_move(self.pdict[self.last_move])

    def save_outcome(self, last_move):
        self.last_move = last_move

class GroupPlayer(object):
    def __init__(self, nplayer, scores, klass=MarkovStrategy, 
                 strategyKwargs={}, **kwargs):
        self.nplayer = nplayer
        self.players = [klass(**strategyKwargs) for i in range(nplayer)]
        self.klass = klass
        self.strategyKwargs = strategyKwargs
    def next_move(self):
        return [p.next_move() for p in self.players]
    def save_outcome(self, outcomes):
        for i,p in enumerate(self.players):
            p.save_outcome(outcomes[i])
    def replicate(self):
        return self.__class__(self.nplayer, None, klass=self.klass,
                              strategyKwargs=self.strategyKwargs)
    def replace(self, i):
        pass
    def is_inference_player(self):
        return False

class InferGroupPlayer(object):
    def __init__(self, nplayer, scores, epsilon=0.05, nwait=10, 
                 initialPval=0.01, optCycles=100, nrecalc=10):
        self.nplayer = nplayer
        self.priorIpLOD = 0.
        self.myIpLOD = numpy.zeros(nplayer)
        self.hisIpLOD = numpy.zeros(nplayer)
        self.players = [InfogainStrategy() for i in range(nplayer)]
        for player in self.players:
            player.nround = 0
        self.nround = 0
        self.scores = scores
        self.nwait = nwait
        self.epsilon = epsilon
        self.initialPval = initialPval
        self.optCycles = optCycles
        self.oldcounts = numpy.zeros(8) # counts from dead group-players
        self.nIpCurrent = -99999 # force initial update
        self.nrecalc = nrecalc
    def next_move(self):
        hisIpP = 1. / (numpy.exp(-self.hisIpLOD) + 1.)
        myIpP = 1. / (numpy.exp(-self.myIpLOD) + 1.)
        moves = []
        for i, p in enumerate(self.players):
            m = p.next_move(self.epsilon, hisP=hisIpP[i], myP=myIpP[i], 
                      optimalStrategy=getattr(self, 'optimalStrategy', None))
            moves.append(m)
        return moves
    def save_outcome(self, outcomes):
        myOdds = numpy.zeros(self.nplayer)
        hisOdds = numpy.zeros(self.nplayer)
        for i,last_move in enumerate(outcomes):
            self.players[i].nround += 1
            if self.players[i].nround == self.nwait: # use groupmax strategy
                self.players[i].__class__ = GroupMaxStrategy
                self.players[i].start(self.myIpLOD[i])
            if hasattr(self, 'groupState'):
                if self.players[i].nround > self.nwait:
                    t = self.players[i].save_outcome(last_move, self.epsilon,
                             state=self.groupState, hisState=self.groupState)
                else: # other player cannot have groupState yet...
                    t = self.players[i].save_outcome(last_move, self.epsilon,
                             state=self.groupState)
            else:
                t = self.players[i].save_outcome(last_move, self.epsilon)
            hisOdds[i] = t[0]
            myOdds[i] = t[1]
        self.hisIpLOD += numpy.log(hisOdds)
        self.myIpLOD += numpy.log(myOdds)
        self.nround += 1
        if self.nround < self.nwait:
            return
        post = 1. / (numpy.exp(self.hisIpLOD) + 1.)
        nIp = (post < self.initialPval).sum() # count confident inf-players
        p = (post.sum() + 1.) / (self.nplayer + 3.) # posterior estimate
        self.priorIpLOD = log((1. - p) / p) # prior odds ratio
        if self.nround == self.nwait: # get group players by p-value
            post2 = numpy.zeros(self.nplayer)
            for i,p in enumerate(self.players):
                if p.match_pval(self.epsilon) < self.initialPval:
                    post2[i] = 1.
            if post2.sum() >= 1.: # found convincing group member(s)
                post = post2
        l = []
        for i,p in enumerate(self.players):
            d = p.state # swap moves to his POV!
            v = numpy.array(d['CC'] + d['DC'] + d['CD'] + d['DD']) * post[i]
            l.append(v)
        counts = numpy.array(l).sum(axis=0) + self.oldcounts
        self.groupState = dict(CC=counts[:2], DC=counts[2:4], 
                               CD=counts[4:6], DD=counts[6:]) # swap back!
        if abs(self.nIpCurrent - nIp) >= self.nrecalc or \
                self.nround % self.optCycles == 0: # update optimal strategy 
            self.nIpCurrent = nIp # save new count
            groupStrategy = (counts[::2] + 1.) / (counts[1::2] + 2.)
            s, r = moran_optimum(nIp, self.nplayer, groupStrategy, 
                                 self.scores, self.epsilon)
            self.optimalStrategy = dict(CC=s[0], CD=s[1], DC=s[2], DD=s[3])
    def replicate(self):
        return self.__class__(self.nplayer, self.scores, self.epsilon, 
                              self.nwait, self.initialPval)
    def replace(self, i):
        'replace player i with new, unknown strategy; restart inference'
        if self.hisIpLOD[i] < log(self.initialPval): # group player
            d = self.players[i].state # swap moves to his POV!
            v = numpy.array(d['CC'] + d['DC'] + d['CD'] + d['DD'])
            self.oldcounts += v # save his counts
        self.players[i] = InfogainStrategy()
        self.players[i].nround = 0
        self.hisIpLOD[i] = self.priorIpLOD # reset posterior odds ratios
        self.myIpLOD[i] = 0.
    def is_inference_player(self):
        return True
        

class InferencePlayer2(object):
    def __init__(self):
        self.state = initial_state()
        self.last_move = None
        self.is_first_move = True

    def first_move(self):
        return 'C'

    def oppenent_conditionals(self):
        q = []
        for move in ['CC','CD','DC','DD']:
            n, N = self.state[move]
            psi = (n + 1.) / (N + 2.) # probability he'll cooperate this round
            q.append(psi)
        return q

    def next_move(self, last_move=None):
        if self.is_first_move:
            self.is_first_move = False
            return self.first_move()
        if last_move:
            if self.last_move: # keep stats on opponent's move
                counts = self.state[self.last_move]
                counts[1] += 1
                if last_move[1] == 'C':
                    counts[0] += 1
            self.last_move = last_move
        q = self.oppenent_conditionals()
        p = calc_best_long_term_response(q)
        r = p[enum[last_move]]
        r2 = random.random()
        if r2 < r:
            return 'C'
        return 'D'
        
    
    #def calc_relent(self, truedict):
        #l = []
        #for k,p in truedict.items():
            #n, N = self.state[swap_moves(k)]
            #l.append(binary_relent(p, (n + 1.) / (N + 2.)))
        #return sum(l)


##################################################################
# basic simulations

class MultiplayerTournament(object):
    def __init__(self, players, epsilon, scores):
        self.players = players
        self.nplayer = nplayer = len(players)
        self.moves = None
        self.epsilon = epsilon
        self.scoresDict = dict(CC=scores[0], CD=scores[1], DC=scores[2],
                               DD=scores[3])
        self.nround = 0
    def do_round(self):
        'run one round of the tournament, and return total score of each player'
        moves = []
        for player in self.players:
            moves.append([move_with_error(m, self.epsilon)
                          for m in player.next_move()])
        scores = []
        for i,player in enumerate(self.players):
            oppMoves = [moves[j][i - 1] for j in range(i)] \
                + [moves[j][i] for j in range(i + 1, self.nplayer)]
            outcomes = [(t[0] + t[1]) for t in zip(moves[i], oppMoves)]
            player.save_outcome(outcomes)
            scores.append(sum([self.scoresDict[g] for g in outcomes]))
        self.nround += 1
        return scores
    def replace(self, die, replicate):
        'replace player i with newPlayer'
        self.players[die] = self.players[replicate].replicate()
        for j,player in enumerate(self.players):
            if j < die:
                player.replace(die - 1)
            elif j > die:
                player.replace(die)
    def report(self, scores):
        s = n = 0.
        for i,p in enumerate(self.players):
            if p.is_inference_player():
                s += scores[i]
                n += 1
        if n > 0 and n < len(scores):
            return n, (s / n) / ((sum(scores) - s) / (len(scores) - n))
    def count_inference_players(self):
        return sum([1 for p in self.players if p.is_inference_player()])

    def fixation_status(self):
        nIp = self.count_inference_players()
        if nIp == 0:
            return False
        elif nIp == self.nplayer:
            return True

def build_tournament(nIp, n, pvec=None, scores=(3,0,5,1), epsilon=0.05):
    if pvec is None:
        pvec = (1., 0., 1., 0.)
    l = [InferGroupPlayer(n - 1, scores, epsilon) for i in range(nIp)]
    while len(l) < n:
        l.append(GroupPlayer(n - 1, scores, strategyKwargs=dict(pvec=pvec)))
    return MultiplayerTournament(l, epsilon, scores)

def moran_selection(scores):
    'get index of player to replicate, player to kill'
    total = sum(scores)
    replicant = random.random() * total
    r = 0.
    for i,s in enumerate(scores):
        r += s
        if replicant <= r:
            break
    return i, random.randrange(len(scores))

def run_tournament(nIp, n, pvec=None, selectionFunction=moran_selection,
                   **kwargs):
    tour = build_tournament(nIp, n, pvec, **kwargs)
    fixed = None
    while fixed is None:
        scores = tour.do_round()
        replicate, die = selectionFunction(scores)
        if replicate != die:
            tour.replace(die, replicate)
        #print tour.report(scores)
        fixed = tour.fixation_status()
    return fixed, tour.nround

def save_tournaments(nIp, nplayer, nmax=100, filename='out.log', **kwargs):
    n = 0
    while n < nmax:
        t = run_tournament(nIp, nplayer, **kwargs)
        print n, t
        if t[0]:
            fixed = 1
        else:
            fixed = 0
        with open(filename, 'a') as ifile:
            print >> ifile, '%d %d' % (fixed, t[1])
        with open(filename, 'r') as ifile:
            n = len(list(ifile))

class Runner(object):
    def __init__(self, *args):
        self.args = args

    def __call__(self, v):
        return run_tournament(*self.args)

def swap_moves(game):
    if game:
        return game[1] + game[0]
    else:
        return None

def move_with_error(m, epsilon):
    if random.random() > epsilon:
        return m
    elif m == 'C':
        return 'D'
    else:
        return 'C'
    

def twoplayer_game(pvec=None, nround=100, last_move = 'CC', epsilon=0.05):
    inferencePlayer = InferencePlayer()
    markovPlayer = MarkovStrategy(pvec)
    l = []
    for i in range(nround):
        Ip = inferencePlayer.calc_relent(markovPlayer.pdict)
        l.append(Ip)
        move1 = inferencePlayer.next_move(last_move)
        move2 = markovPlayer.next_move(swap_moves(last_move))
        last_move = move_with_error(move1, epsilon) \
            + move_with_error(move2, epsilon)
    return l, inferencePlayer, markovPlayer


def time_to_identify(pvec, pId=.001, klass=InfogainStrategy, epsilon=0.05):
    n = 0
    inferencePlayer = klass()
    markovPlayer = MarkovStrategy(pvec)
    myP = hisP = 1.
    while inferencePlayer.match_pval(epsilon) > pId:
        move1 = inferencePlayer.next_move(epsilon)
        move2 = markovPlayer.next_move(epsilon)
        last_move = move_with_error(move1, epsilon) \
            + move_with_error(move2, epsilon)
        t = inferencePlayer.save_outcome(last_move)
        hisP *= t[0]
        myP *= t[1]
        print inferencePlayer.myIpMoves[-1] + inferencePlayer.hisIpMoves[-1],\
            move1 + move2, last_move, (myP, hisP), \
            inferencePlayer.match_pval(epsilon)
            
        markovPlayer.save_outcome(swap_moves(last_move))
        n += 1
    return n

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

#def stationary_dist(t, epsilon=1e-10):
    #'compute stationary dist from transition matrix'
    #diff = 1.
    #while diff > epsilon:
        #t = linalg.matrix_power(t, 2)
        #w = t.sum(axis=1) # get row sums
        #t /= w.reshape((len(w), 1)) # normalize each row
        #m = numpy.mean(t, axis=0)
        #diff = numpy.dot(m, t) - m
        #diff = (diff * diff).sum()
    #return m

#def stationary_dist2(t, epsilon=.001):
    #'compute stationary distribution using eigenvector method'
    #w, v = linalg.eig(t.transpose())
    #for i,eigenval in enumerate(w):
        #s = numpy.real_if_close(v[:,i]) # handle small complex number errors
        #s /= s.sum() # normalize
        #if abs(eigenval - 1.) <= epsilon and (s >= 0.).sum() == len(s):
            #return s # must have unit eigenvalue and all non-neg components
    #raise ValueError('no stationary eigenvalue??')

#def game_transition_matrix(myProbs, hisProbs0):
    #'compute transition rate matrix for a strategy pair'
    # have to swap moves for other player...
    #hisProbs = (hisProbs0[0], hisProbs0[2],hisProbs0[1], hisProbs0[3])
    #l = []
    #for i,myP in enumerate(myProbs):
        #hisP = hisProbs[i]
        #l.append((myP * hisP, myP * (1. - hisP), 
                  #(1. - myP) * hisP, (1. - myP) * (1. - hisP)))
    #return numpy.array(l)

#def stationary_rates(myProbs, hisProbs):
    #'compute expectation rates of all possible transitions for strategy pair'
    #t = game_transition_matrix(myProbs, hisProbs)
    #s = stationary_dist2(t)
    #return [p * t[i] for (i,p) in enumerate(s)]

#def stationary_score(myProbs, hisProbs, scores):
    #'compute expectation score for my strategy vs. opponent strategy'
    #rates = stationary_rates(myProbs, hisProbs)
    #l = [scores * vec for vec in rates]
    #return numpy.array(l).sum()

def stationary_score(myProbs, hisProbs, scores):
    'compute expectation score for my strategy vs. opponent strategy'
    s = exact_stationary(myProbs, hisProbs)
    return numpy.dot(s, numpy.array(scores))

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

def reciprocal_scores(myProbs, hisProbs, scores):
    sAB = stationary_score(myProbs, hisProbs, scores)
    sBA = stationary_score(hisProbs, myProbs, scores)
    return sAB, sBA

def self_scores(hisProbs, scores, epsilon=0.05):
    allC = (1. - epsilon,1. - epsilon,1. - epsilon,1. - epsilon,)
    sAA = stationary_score(allC, allC, scores)
    sBB = stationary_score(hisProbs, hisProbs, scores)
    return sAA, sBB

def population_score_diff(myFrac, hisFrac, myProbs, hisProbs, scores):
    sAB, sBA = reciprocal_scores(myProbs, hisProbs, scores)
    return hisFrac * sAB - myFrac * sBA

def population_optimum2(myFrac, hisFrac, hisProbs, scores, epsilon=0.05):
    'find optimal strategy at the specified population fraction'
    s = optimize.fmin_tnc(lambda myProbs: -1* population_score_diff(myFrac, hisFrac, add_noise_vector(myProbs, epsilon), hisProbs, scores), 
                          [0.5,0.5,0.5,0.5], bounds=[(0,1)]*4, 
                          approx_grad=True, messages=0, maxfun=1000)[0]
    return s, population_diff(myFrac, add_noise_vector(s, epsilon), 
                              hisProbs, scores, hisFrac, epsilon)

def fitness_ratio(myFrac, hisFrac, myProbs, hisProbs, scores, myBase, hisBase):
    sAB, sBA = reciprocal_scores(myProbs, hisProbs, scores)
    return (myBase + hisFrac * sAB) / (hisBase + myFrac * sBA)

def moran_optimum(m, n, hisProbs, scores, epsilon=0.05, start=None):
    'find strategy that maximizes fitness ratio vs. opponent'
    if start is None:
        start = (0.5, 0.5, 0.5, 0.5)
    sAA, sBB = self_scores(hisProbs, scores, epsilon)
    myBase = sAA * float(m) / n
    hisFrac = float(n - m) / n
    hisBase = sBB * float(n - m - 1) / n
    myFrac = float(m + 1) / n
    s = optimize.fmin_tnc(lambda myProbs: 
                          -fitness_ratio(myFrac, hisFrac,
                                         add_noise_vector(myProbs, epsilon), 
                                         hisProbs, scores, myBase, hisBase),
                          start, bounds=[(0,1)]*4, 
                          approx_grad=True, messages=0, maxfun=1000)[0]
    return s, fitness_ratio(myFrac, hisFrac, add_noise_vector(s, epsilon), 
                            hisProbs, scores, myBase, hisBase)

def population_diff(myFrac, myProbs, hisProbs, scores, hisFrac=None, 
                    epsilon=0.05):
    'compute relative score for strategy pair at specified population fraction'
    if hisFrac is None:
        hisFrac = 1. - myFrac
    e_ = 1. - epsilon
    sAA = e_ * e_ * scores[0] + e_ * epsilon * (scores[1] + scores[2]) \
        + epsilon * epsilon * scores[3]
    sBB = stationary_score(hisProbs, hisProbs, scores)
    sAB = stationary_score(myProbs, hisProbs, scores)
    sBA = stationary_score(hisProbs, myProbs, scores)
    return myFrac * sAA + hisFrac * sAB \
        - hisFrac * sBB - myFrac * sBA 

def zd_vector1(chi):
    return (1. - (2. * chi - 2.) / (4. * chi + 1.), 0.,
            (chi + 4.) / (4. * chi + 1.), 0.)

def zd_vector2(chi):
    return (1., (chi - 1.)/(3. * chi + 2.), 1., 2.*(chi - 1.)/(3. * chi + 2.))

def add_noise(p, epsilon):
    return p * (1. - epsilon) + (1. - p) * epsilon

def add_noise_vector(l, epsilon):
    return [add_noise(p, epsilon) for p in l]

if __name__ == '__main__':
    import sys
    args = [int(s) for s in sys.argv[1:4]] + sys.argv[4:]
    save_tournaments(*args)
