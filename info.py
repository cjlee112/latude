import random
from scipy import stats, optimize
from math import log, exp, sqrt
from numpy import linalg
import numpy
import functools
import hashlib
import players

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
    p, _, _ = optimize.fmin_tnc(lambda p: -stationary_scores(p, q, scores)[0], 
                                [0.5,0.5,0.5,0.5], bounds=[(0,1)]*4, 
                                approx_grad=True, messages=0, maxfun=1000)
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
        else: # no history, so use uninformative prior of 0.5
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
    _inGroupMax = True
    def start(self, myIpLOD=0., tau=0.01):
        self.pPartner = 1. / (exp(-myIpLOD) + 1.)
        self.tau = tau
        self.last_move2 = self.last_move # keep history for 2 moves...
    def next_move(self, epsilon=0.05, hisP=0, myP=0, optimalStrategy=None, selfStrategy=None):
        if hisP >= 0.5: # cooperate with any inference player
            s = selfStrategy
            self.isFoe = False
        else: # apply optimal strategy vs. group
            s = optimalStrategy
            self.isFoe = True
        self.myIpPC = add_noise(s[self.last_move], epsilon)
        myMove = stochastic_move(s[self.last_move])
        self.update_hmm(epsilon, optimalStrategy, selfStrategy)
        return myMove
    def update_hmm(self, epsilon, optimalStrategy, selfStrategy):
        '2-state HMM for whether inf-player is partnering with us or not'
        ctx = swap_moves(self.last_move2)
        p1 = ((1. - self.tau) * self.pPartner 
              + self.tau * (1. - self.pPartner)) \
              * move_likelihood(add_noise(selfStrategy[ctx], epsilon), 
                                self.last_move[1]) # partner
        p2 = (self.tau * self.pPartner 
              + (1. - self.tau) * (1. - self.pPartner)) \
              * move_likelihood(add_noise(optimalStrategy[ctx], epsilon),
                                self.last_move[1]) # not partnering with me
        self.pPartner = p = p1 / (p1 + p2)
        ctx = swap_moves(self.last_move)
        pC = p * selfStrategy[ctx] + (1. - p) * optimalStrategy[ctx]
        self.hisIpPC = add_noise(pC, epsilon) # p(inf-player cooperates w/ me)
        self.last_move2 = self.last_move

class MarkovStrategy(object):
    def __init__(self, pvec=None, firstMoveP=None):
        if not pvec:
            pvec = [random.random() for i in range(4)]
        self.pdict = dict(CC=pvec[0], CD=pvec[1], DC=pvec[2], DD=pvec[3])
        if firstMoveP is None: # take average of 4D probabilities
            firstMoveP = float(sum(pvec)) / len(pvec)
        self.firstMoveP = firstMoveP
        self.last_move = None

    def next_move(self, epsilon=0.05):
        if not self.last_move:
            return stochastic_move(self.firstMoveP)
        return stochastic_move(self.pdict[self.last_move])

    def save_outcome(self, last_move):
        self.last_move = last_move

class GroupPlayer(object):
    def __init__(self, nplayer, scores, klass=MarkovStrategy, 
                 strategyKwargs={}, name='M', **kwargs):
        self.nplayer = nplayer
        self.players = [klass(**strategyKwargs) for i in range(nplayer)]
        self.klass = klass
        self.strategyKwargs = strategyKwargs
        self.name = name
    def next_move(self):
        return [p.next_move() for p in self.players]
    def save_outcome(self, outcomes):
        for i,p in enumerate(self.players):
            p.save_outcome(outcomes[i])
    def replicate(self):
        return self.__class__(self.nplayer, None, klass=self.klass,
                              strategyKwargs=self.strategyKwargs, 
                              name=self.name)
    def replace(self, i):
        pass
    def is_inference_player(self):
        return False

class InferGroupPlayer(object):
    _initialStrategy = InfogainStrategy
    def __init__(self, nplayer, scores, epsilon=0.05, nwait=10, 
                 initialPval=0.01, optCycles=100, nrecalc=10, name='I'):
        self.nplayer = nplayer
        self.priorIpLOD = 0.
        self.myIpLOD = numpy.zeros(nplayer)
        self.hisIpLOD = numpy.zeros(nplayer)
        self.players = [self._initialStrategy() for i in range(nplayer)]
        for player in self.players:
            player.nround = 0
        self.nround = 0
        self.scores = scores
        self.nwait = nwait
        if epsilon == 0.: # prevent numeric underflow
            epsilon = 1e-6
        self.epsilon = epsilon
        self.initialPval = initialPval
        self.optCycles = optCycles
        self.oldcounts = numpy.zeros(8) # counts from dead group-players
        self.nIpCurrent = -99999 # force initial update
        self.nrecalc = nrecalc
        self.name = name
        self.optimizer = StrategyOptimizer(scores)
    def next_move(self):
        hisIpP = 1. / (numpy.exp(-self.hisIpLOD) + 1.)
        myIpP = 1. / (numpy.exp(-self.myIpLOD) + 1.)
        moves = []
        for i, p in enumerate(self.players):
            m = p.next_move(self.epsilon, hisP=hisIpP[i], myP=myIpP[i], 
                      optimalStrategy=getattr(self, 'optimalStrategy', None),
                            selfStrategy=self.optimizer.selfStrategy)
            moves.append(m)
        return moves
    def save_outcome(self, outcomes, optFunc=None):
        if optFunc is None:
            optFunc = moran_optimum
        myOdds = numpy.zeros(self.nplayer)
        hisOdds = numpy.zeros(self.nplayer)
        doGroupMax = False
        for i,last_move in enumerate(outcomes):
            self.players[i].nround += 1
            if self.do_groupmax(i):
                doGroupMax = True
                if not getattr(self.players[i], '_inGroupMax', False):
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
        if not doGroupMax:
            return
        post = 1. / (numpy.exp(self.hisIpLOD) + 1.)
        nIp = (post < self.initialPval).sum() # count confident inf-players
        p = (post.sum() + 1.) / (self.nplayer + 3.) # posterior estimate
        self.priorIpLOD = log((1. - p) / p) # prior odds ratio
        post = self.get_group_weights(post)
        counts = self.get_group_counts(post)
        self.groupState = dict(CC=counts[:2], DC=counts[2:4], 
                               CD=counts[4:6], DD=counts[6:]) # swap to my POV
        self.update_strategy(nIp, counts, outcomes, optFunc)
    def update_strategy(self, nIp, counts, outcomes, 
                        optFunc=None, countsMin=1000):
        groupOutcomes = [o for (i,o) in enumerate(outcomes)
                         if getattr(self.players[i], 'isFoe', False)]
        s = self.optimizer.update_strategy(groupOutcomes, counts, nIp,
                                           self.nplayer, self.epsilon)
        if s:
            self.optimalStrategy = dict(CC=s[0], CD=s[1], DC=s[2], 
                                        DD=s[3])
    def do_groupmax(self, i):
        return self.players[i].nround >= self.nwait
    def get_group_scores(self, outcomes):
        'get my average score, his average score, count vs. the group'
        myScore = hisScore = n = 0
        for i,p in enumerate(self.players):
            if getattr(p, 'isFoe', False):
                myScore += self.scores[outcomes[i]] 
                hisScore += self.scores[swap_moves(outcomes[i])]
                n += 1
        if n > 0:
            return float(myScore) / n, float(hisScore) / n, n
    def get_group_weights(self, post, filterFunc=None):
        if filterFunc is None and self.nround == self.nwait: # get group players by p-value
            filterFunc = lambda x:numpy.array(
                [(p.match_pval(self.epsilon) < self.initialPval) 
                 for p in self.players])
        if filterFunc: # apply filter function
            post2 = numpy.zeros(self.nplayer)
            post2[filterFunc(post)] = 1.
            if post2.sum() >= 1.: # found convincing group member(s)
                return post2
        return post
    def get_group_counts(self, pGroup):
        l = []
        for i,p in enumerate(self.players):
            d = p.state # swap moves to his POV!
            v = numpy.array(d['CC'] + d['DC'] + d['CD'] + d['DD']) * pGroup[i]
            l.append(v)
        counts = numpy.array(l).sum(axis=0) + self.oldcounts
        return counts
    def replicate(self):
        return self.__class__(self.nplayer, self.scores, self.epsilon, 
                              self.nwait, self.initialPval, name=self.name)
    def replace(self, i):
        'replace player i with new, unknown strategy; restart inference'
        if self.hisIpLOD[i] < log(self.initialPval): # group player
            d = self.players[i].state # swap moves to his POV!
            v = numpy.array(d['CC'] + d['DC'] + d['CD'] + d['DD'])
            self.oldcounts += v # save his counts
        self.players[i] = self._initialStrategy()
        self.players[i].nround = 0
        self.hisIpLOD[i] = self.priorIpLOD # reset posterior odds ratios
        self.myIpLOD[i] = 0.
    def is_inference_player(self):
        return True
    def check_accuracy(self, names):
        matched = [(name == self.name) for name in names]
        groupmax = [isinstance(p, GroupMaxStrategy) for p in self.players]
        prediction = (self.hisIpLOD > 0.)
        d = {}
        for i,p in enumerate(prediction):
            if matched[i] != p:
                d[p] = d.get(p, 0) + 1
                d[groupmax[i], p] = d.get((groupmax[i], p), 0) + 1
        return d
        
class InferGroupPlayer2(InferGroupPlayer):
    'maximizes difference in scores rather than ratio'
    def save_outcome(self, outcomes):
        return InferGroupPlayer.save_outcome(self, outcomes, diff_optimum)

class InferGroupPlayerZeroNoise(InferGroupPlayer2):
    'for use with epsilon=0'
    def report_mismatches(self, post):
        'report non-inf players'
        return numpy.array([(p.mismatches > 0) for p in self.players])
    def get_group_weights(self, post):
        return InferGroupPlayer2.get_group_weights(self, post, 
                                                   self.report_mismatches)

class TagStrategy(GroupMaxStrategy):
    hisIpPC = myIpPC = 0.5
    def update_hmm(self, epsilon, optimalStrategy):
        pass

class InferGroupPlayerTags(InferGroupPlayer):
    'positive control: use true identities to get optimal strategy'
    _initialStrategy = TagStrategy
    def save_outcome(self, outcomes, tags=None):
        self.nround += 1
        for i,last_move in enumerate(outcomes):
            self.players[i].save_outcome(last_move, self.epsilon)
        post = numpy.zeros(self.nplayer)
        nIp = 0
        for i,name in enumerate(tags):
            if name != self.name:
                post[i] = 1.
                self.hisIpLOD[i] = -999.
            else:
                self.hisIpLOD[i] = 999.
                nIp += 1
        counts = self.get_group_counts(post)
        self.groupState = dict(CC=counts[:2], DC=counts[2:4], 
                               CD=counts[4:6], DD=counts[6:]) # swap to my POV
        if self.nround > 1:
            self.update_strategy(nIp, counts)
        else:
            self.optimalStrategy = dict(CC=1., CD=1., DC=1., DD=1.)
        
    

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
    def save_outcome(self, i, outcomes):
        return self.players[i].save_outcome(outcomes)
    def do_round(self):
        'run one round of the tournament, and return total score of each player'
        moves = []
        for player in self.players:
            moves.append([move_with_error(m, self.epsilon)
                          for m in player.next_move()])
        scores = []
        for i in range(self.nplayer):
            oppMoves = [moves[j][i - 1] for j in range(i)] \
                + [moves[j][i] for j in range(i + 1, self.nplayer)]
            outcomes = [(t[0] + t[1]) for t in zip(moves[i], oppMoves)]
            self.save_outcome(i, outcomes)
            scores.append(sum([self.scoresDict[g] for g in outcomes])
                          / float(len(outcomes))) # player's average score
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
        d = {}
        n = nInfogain = 0
        for i,p in enumerate(self.players):
            try:
                l = d[p.name]
            except KeyError:
                l = d[p.name] = [0, 0]
            l[0] += scores[i]
            l[1] += 1.
            if isinstance(p, InferGroupPlayer):
                n += len(p.players)
                for ps in p.players:
                    if not getattr(ps, '_inGroupMax', False):
                        nInfogain += 1
        results = {'infogain':float(nInfogain) / n}
        for k,v in d.items():
            results[k] = v[0] / v[1]
            results['#' + k] = v[1]
        return results

    def get_player_names(self):
        return [p.name for p in self.players]

    def check_accuracy(self, *args):
        names = self.get_player_names()
        n = 0
        d = {}
        for i, name in enumerate(names):
            if name == 'I': 
                for k,v in self.players[i].check_accuracy(names[:i] + names[i + 1:]).items():
                    d[k] = d.get(k, 0) + v
                n += len(self.players) - 1
        for k, v in d.items():
            d[k] = float(v) / n
        return d

    def count_players(self):
        d = {}
        for p in self.players:
            d[p.name] = d.get(p.name, 0) + 1
        return d

    def fixation_status(self):
        d = self.count_players()
        if len(d) > 1:
            return False
        else:
            return d.keys()[0]

class MultiplayerTournament2(MultiplayerTournament):
    'do not reset player unless replaced by a different type'
    def replace(self, die, replicate):
        if self.players[die].name == self.players[replicate].name:
            return # if same type, do nothing
        return MultiplayerTournament.replace(self, die, replicate)

class MultiplayerTournamentTags(MultiplayerTournament2):
    def save_outcome(self, i, outcomes):
        tags = [p.name for (j,p) in enumerate(self.players) if j != i]
        try: # pass tags if player will use them
            return self.players[i].save_outcome(outcomes, tags=tags)
        except TypeError: # default: player does not accept tags
            return self.players[i].save_outcome(outcomes)

def build_tournament(nIp, n, pvec=None, scores=(3,0,5,1), epsilon=0.05,
                     klass=InferGroupPlayer, 
                     tournamentClass=MultiplayerTournament, **kwargs):
    if pvec is None:
        pvec = (1., 0., 1., 0.)
    l = [klass(n - 1, scores, epsilon) for i in range(nIp)]
    while len(l) < n:
        l.append(GroupPlayer(n - 1, scores, strategyKwargs=dict(pvec=pvec),
                             **kwargs))
    return tournamentClass(l, epsilon, scores)

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

def exp_imitation(scores, beta=1.):
    i = j = random.randrange(len(scores))
    while i == j:
        j = random.randrange(len(scores))
    x1 = exp(beta * scores[i])
    x2 = exp(beta * scores[j])
    if random.random() <= x1 / (x1 + x2):
        return i, j # replace j by i
    else:
        return j, i # replace i by j

def run_tournament(nIp, n, pvec=None, selectionFunction=moran_selection,
                   selectionPeriod=1, **kwargs):
    tour = build_tournament(nIp, n, pvec, **kwargs)
    fixed = False
    i = 0
    while not fixed:
        scores = tour.do_round()
        i += 1
        if i % selectionPeriod: # only apply selection once per selectionPeriod
            continue
        replicate, die = selectionFunction(scores)
        if replicate != die:
            tour.replace(die, replicate)
        fixed = tour.fixation_status()
    return fixed, tour.nround

def save_tournaments(nIp, nplayer, nmax=100, filename='out.log', 
                     useFileSize=True, **kwargs):
    n = 0
    while n < nmax:
        t = run_tournament(nIp, nplayer, **kwargs)
        with open(filename, 'a') as ifile:
            print >> ifile, '%s %d' % t
        if useFileSize: # count total number of completed runs from logfile
            with open(filename, 'r') as ifile:
                n = len(list(ifile))
        else: # just perform nmax runs
            n += 1

def check_accuracy(nIp, n, pvec=None, ncycle=100, klass=InferGroupPlayer2,
                   epsilon=0.05, tournamentClass=MultiplayerTournament2,
                   scores=(2, -1, 3, 0), methodName='report'):
    if epsilon == 0.:
        klass = InferGroupPlayerZeroNoise
    tour = build_tournament(nIp, n, pvec, klass=klass, epsilon=epsilon,
                            tournamentClass=tournamentClass, scores=scores)
    l = []
    reportMethod = getattr(tour, methodName)
    for i in range(ncycle):
        scores = tour.do_round()
        l.append(reportMethod(scores))
        #print '#I:', l[-1]['#I']
        replicate, die = exp_imitation(scores)
        if replicate != die:
            tour.replace(die, replicate)
        if tour.fixation_status():
            break
    return l, tour

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
    if n == 0.:
        raise ValueError('exact_stationary() cannot handle zeros')
    s = numpy.array(s) / n
    return s

def stationary_dist(myProbs, hisProbs, epsilon=1e-10):
    'compute stationary dist from transition matrix'
    t = game_transition_matrix(myProbs, hisProbs)
    diff = 1.
    while diff > epsilon:
        t = linalg.matrix_power(t, 2)
        w = t.sum(axis=1) # get row sums
        t /= w.reshape((len(w), 1)) # normalize each row
        m = numpy.mean(t, axis=0)
        diff = numpy.dot(m, t) - m
        diff = (diff * diff).sum()
    return m

def stationary_dist2(myProbs, hisProbs, epsilon=.001):
    'compute stationary distribution using eigenvector method'
    t = game_transition_matrix(myProbs, hisProbs)
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

def stationary_scores(myProbs, hisProbs, scores):
    'compute expectation score for my strategy vs. opponent strategy'
    try:
        s = exact_stationary(myProbs, hisProbs)
    except ValueError:
        s = stationary_dist(myProbs, hisProbs)
    return ((scores * s).sum(), 
            ((scores[0], scores[2], scores[1], scores[3]) * s).sum())

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
        l.append((stationary_scores(myProbs, hisProbs, scores)[0], myProbs))
    l.sort()
    return l

def all_vs_all(scores, **kwargs):
    'rank all possible strategies by their minimum score vs. all strategies'
    l = []
    for myProbs in generate_corners(**kwargs):
        l.append((min([(stationary_scores(myProbs, hisProbs, scores)[0], hisProbs)
                       for hisProbs in generate_corners(**kwargs)]), myProbs))
    l.sort()
    return l

def population_optimum(myFrac, hisFrac, hisProbs, scores, epsilon=0.05):
    'find optimal corner strategy at the specified population fraction'
    l = []
    for myProbs in generate_corners(epsilon):
        sAB, sBA = stationary_scores(myProbs, hisProbs, scores)
        l.append((hisFrac * sAB - myFrac * sBA, myProbs))
    l.sort()
    return l[-1]

def self_scores(hisProbs, scores, epsilon=0.05):
    allC = (1. - epsilon,1. - epsilon,1. - epsilon,1. - epsilon,)
    sAA = stationary_scores(allC, allC, scores)[0]
    sBB = stationary_scores(hisProbs, hisProbs, scores)[0]
    return sAA, sBB

def population_score_diff(myFrac, hisFrac, myProbs, hisProbs, scores):
    sAB, sBA = stationary_scores(myProbs, hisProbs, scores)
    return hisFrac * sAB - myFrac * sBA

def population_optimum2(myFrac, hisFrac, hisProbs, scores, epsilon=0.05):
    'find optimal strategy at the specified population fraction'
    s = optimize.fmin_tnc(lambda myProbs: -1* population_score_diff(myFrac, hisFrac, add_noise_vector(myProbs, epsilon), hisProbs, scores), 
                          [0.5,0.5,0.5,0.5], bounds=[(0,1)]*4, 
                          approx_grad=True, messages=0, maxfun=1000)[0]
    return s, population_diff(myFrac, add_noise_vector(s, epsilon), 
                              hisProbs, scores, hisFrac, epsilon)

def fitness_ratio(myFrac, hisFrac, myProbs, hisProbs, scores, myBase, hisBase):
    sAB, sBA = stationary_scores(myProbs, hisProbs, scores)
    return (myBase + hisFrac * sAB) / (hisBase + myFrac * sBA)

def moran_optimum(m, n, hisProbs, scores, epsilon=0.05, start=None, 
                  negFunc=None):
    'find strategy that maximizes fitness ratio vs. opponent'
    if start is None:
        start = (0.5, 0.5, 0.5, 0.5)
    sAA, sBB = self_scores(hisProbs, scores, epsilon)
    myBase = sAA * float(m) / n
    hisFrac = float(n - m) / n
    hisBase = sBB * float(n - m - 1) / n
    myFrac = float(m + 1) / n
    if negFunc is None:
        negFunc = lambda myProbs: \
            -fitness_ratio(myFrac, hisFrac, add_noise_vector(myProbs, epsilon), 
                           hisProbs, scores, myBase, hisBase)
    s = optimize.fmin_tnc(negFunc, start, bounds=[(0,1)]*4, 
                          approx_grad=True, messages=0, maxfun=1000)[0]
    return s, -negFunc(s)

def diff_optimum(m, n, hisProbs, scores, epsilon=0.05, **kwargs):
    'find strategy that maximizes score difference'
    hisFrac = float(n - m) / n
    myFrac = float(m + 1) / n
    negFunc = lambda myProbs: \
        -population_score_diff(myFrac, hisFrac, 
                               add_noise_vector(myProbs, epsilon),
                               hisProbs, scores)
    return moran_optimum(m, n, hisProbs, scores, epsilon, negFunc=negFunc, 
                         **kwargs)

def optimize_selfscore(scores, epsilon=0.05, start=(1., 1., 1., 1.)):
    'find strategy that maximizes its self-score'
    def neg_selfscore(pvec):
        pvec = add_noise_vector(pvec, epsilon)
        return -stationary_scores(pvec, pvec, scores)[0]
    return optimize.fmin_tnc(neg_selfscore, start, bounds=[(0,1)]*4, 
                             approx_grad=True, messages=0, maxfun=1000)[0]

def population_diff(myFrac, myProbs, hisProbs, scores, hisFrac=None, 
                    epsilon=0.05):
    'compute relative score for strategy pair at specified population fraction'
    if hisFrac is None:
        hisFrac = 1. - myFrac
    e_ = 1. - epsilon
    sAA = e_ * e_ * scores[0] + e_ * epsilon * (scores[1] + scores[2]) \
        + epsilon * epsilon * scores[3]
    sBB = stationary_scores(hisProbs, hisProbs, scores)[0]
    sAB, sBA = stationary_scores(myProbs, hisProbs, scores)
    return myFrac * sAA + hisFrac * sAB \
        - hisFrac * sBB - myFrac * sBA 

def sample_posterior_strategies(counts, epsilon, nsample=100):
    'return sample of strategy vectors drawn from posterior distribution'
    l = []
    for i in range(0, 8, 2):
        rv = stats.beta(counts[i] + 1, counts[i + 1] - counts[i] + 1)
        data = rv.rvs(nsample)
        if epsilon > 0.: # truncate to accessible region
            data[data < epsilon] = epsilon
            data[data > 1. - epsilon] = 1. - epsilon
        l.append(data)
    return zip(*l)
    
def get_sample_scores(myProbs, pvecSample, scores):
    return [stationary_scores(myProbs, pvec, scores) for pvec in pvecSample]

def get_line_params(scoreSample, n):
    return [(t[0] + t[1], n * t[0] - t[1]) for t in scoreSample]

def get_intersections(lineSample1, lineSample2):
    l = []
    for i,t in enumerate(lineSample1):
        l.append((t[1] - lineSample2[i][1]) / (t[0] - lineSample2[i][0]))
    l.sort()
    return l

def choose_allc_vs_alld(counts, m, n, scores, coop=(1., 1., 1., 1.), 
                        epsilon=0.05, p=0.05, nsample=100):
    alld = (0., 0., 0., 0.)
    pvecSample = sample_posterior_strategies(counts, epsilon, nsample)
    scoreSampleC = get_sample_scores(coop, pvecSample, scores)
    scoreSampleD = get_sample_scores(alld, pvecSample, scores)
    lineSampleC = get_line_params(scoreSampleC, n)
    lineSampleD = get_line_params(scoreSampleD, n)
    mSample = get_intersections(lineSampleC, lineSampleD)
    i = int(p * len(mSample))
    if m < mSample[-i - 1] + 20:
        best = coop
    else:
        best = alld
    return best, mSample[i], mSample[-i - 1]

class StrategyDict(dict):
    'allow approximate matches to handle slight variation in optima'
    def __init__(self, digits=3):
        self.digits = digits
        dict.__init__(self)
    def approx_key(self, k):
        return tuple([round(v, self.digits) for v in k])
    def __getitem__(self, k):
        return dict.__getitem__(self, self.approx_key(k))
    def __setitem__(self, k, v):
        return dict.__setitem__(self, self.approx_key(k), v)
        


class StrategyOptimizer(object):
    def __init__(self, scores, epsilon=0.05, tolerance=0.05, conf=0.9,
                 candidates=(players.allc, players.tft, players.wsls, 
                             players.zdr2, players.alld)):
        self.scores = scores
        self.tolerance = tolerance
        self.conf = conf
        self.strategies = StrategyDict()
        for pvec in candidates:
            self.strategies[pvec] = StrategyScorer(pvec, scores)
        self.current = None
        self.nrefresh = 0
        s = optimize_selfscore(self.scores, epsilon)
        self.selfStrategy = dict(CC=s[0], CD=s[1], DC=s[2], DD=s[3])
    def save_optimal_strategy(self, groupStrategy, m, n, epsilon):
        'search for optimal strategy and add to self.strategies if new'
        s = diff_optimum(m, n, groupStrategy, self.scores, epsilon)[0]
        try:
            self.strategies[s]
        except KeyError: # new strategy, so add it to our options
            self.strategies[s] = StrategyScorer(s, self.scores)
    def update_sampling(self, counts, n, epsilon, nsample=200):
        'draw posterior score dist for each of our possible strategies'
        groupStrategy = (counts[::2] + 1.) / (counts[1::2] + 2.)
        self.save_optimal_strategy(groupStrategy, 0, n, epsilon)
        self.save_optimal_strategy(groupStrategy, n - 1, n, epsilon)
        pvecSample = sample_posterior_strategies(counts, epsilon, 
                                                 nsample)
        for pvec, strategyScorer in self.strategies.items():
            strategyScorer.compute_scores(pvecSample, epsilon)
        self.minorityStrategy = self.assess_confidence(0, n)[-1]
        self.majorityStrategy = self.assess_confidence(n - 1, n)[-1]
    def update_strategy(self, outcomes, counts, m, n, epsilon, 
                        nsample=200):
        #if self.current in self.strategies:
        #    self.strategies[self.current].save_outcomes(outcomes)
        if not self.need_refresh(m, n):
            return None
        print 'update', m, self.nrefresh
        self.update_sampling(counts, n, epsilon, nsample)
        self.transition = self.compute_transition(n)
        candidate = self.get_candidate(m)
        if candidate == self.current:
            return None
        elif not self.current:
            return self.start_strategy(candidate, outcomes)
        if m == 0: 
            conf = self.minorityStrategy[0]
        elif m == n - 1:
            conf = self.majorityStrategy[0]
        else:
            conf = [t[0] for t in self.assess_confidence(m, n)
                    if t[1] == candidate][0]
        if conf >= self.conf: # confidently better, so switch strategy
            return self.start_strategy(candidate, outcomes)
    def get_candidate(self, m, p=0.5):
        if not self.transition: # minor == major
            return self.minorityStrategy[1] # one pvec to rule them all
        i = int(p * len(self.transition))
        if m < self.transition[i]:
            return self.minorityStrategy[1]
        else:
            return self.majorityStrategy[1]
    def assess_confidence(self, m, n):
        'get sorted list of (conf, s) for all relevant strategies'
        d = {}
        for pvec, strategyScorer in self.strategies.items():
            d[pvec] = deltas = strategyScorer.compute_deltas(m, n)
            nsample = len(deltas)
        maxDelta = [max([deltas[i] for deltas in d.values()])
                    for i in range(nsample)]
        l = []
        for pvec, deltas in d.items():
            nbest = sum([1 for i in range(nsample) 
                         if deltas[i] >= maxDelta[i] - self.tolerance])
            l.append((nbest / float(len(deltas)), pvec))
        l.sort()
        return l
    def compute_transition(self, n):
        minor = self.minorityStrategy[1]
        major = self.majorityStrategy[1]
        if minor == major:
            return None
        minorSample = get_line_params(self.strategies[minor].scoreSample, n)
        majorSample = get_line_params(self.strategies[major].scoreSample, n)
        return get_intersections(minorSample, majorSample)
    def start_strategy(self, best, outcomes):
        print 'STRATEGY:', best
        self.current = best
        self.nrefresh = 0
        self.strategies[best].start_record(len(outcomes))
        return best
    def need_refresh(self, m, n, p=0.5):
        self.nrefresh += 1
        if not self.current:
            return True
        if m <= n / 2:
            if self.minorityStrategy[0] < self.conf:
                return True # need to update minority strategy
        elif self.majorityStrategy[0] < self.conf:
            return True # need to update majority strategy
        if self.nrefresh < 20: # monitor new strategy closely
            return True
        if not self.transition or m < self.transition[0] or \
                m > self.transition[-1]: # not in transition zone
            return False
        if getattr(self, 'zoneCheck', True): # 1st time in zone?
            self.zoneCheck = False
            return True # force an update
        i = int(p * len(self.transition))
        if self.current == self.minorityStrategy[1]:
            return m > self.transition[i] # nearing transition to major
        else:
            return m < self.transition[i] # nearing transition to minor

        

class StrategyScorer(object):
    'predict distribution of stationary score for a specific strategy'
    def __init__(self, pvec, scores):
        self.pvec = pvec
        self.scores = scores
        self.empiricalData = []
    def compute_scores(self, pvecSample, epsilon):
        'save (sAB, sBA) for me vs. each model in pvecSample'
        s = add_noise_vector(self.pvec, epsilon)
        self.scoreSample = get_sample_scores(s, pvecSample, self.scores)
    def compute_deltas(self, m, n):
        'mean, lower & upper bound estimates for score delta'
        fA, fB = get_population_fractions(m, n)
        return [(fB * t[0] - fA * t[1]) for t in self.scoreSample]
    def predict_delta(self, m, n, p=0.025):
        l = self.compute_deltas(m, n)
        l.sort()
        i = int(p * len(l))
        self.mean = sum(l) / len(l)
        self.lower = l[i]
        self.upper = l[-i - 1]
        #print 'predict %s %1.3f %1.3f %1.3f' \
        #    % (str(self.pvec), self.mean, self.lower, self.upper)
        return self.mean, self.lower, self.upper
    def best_estimate(self, m, n):
        if not self.empiricalData:
            return self.predict_delta(m, n)[0]
        scoreSequence = self.get_best_score_sequence()
        delta = choose_empirical_vs_predicted_mean(scoreSequence, self, 
                                                   m, n)
        self.mean = delta
        #print '\tBEST %1.3f' % delta
        return delta
    def get_best_score_sequence(self):
        l = [(len(ss.data), ss) for ss in self.empiricalData]
        l.sort()
        return l[-1][1] # longest ScoreSequence
    def start_record(self, ngroup):
        self.empiricalData.append(ScoreSequence(ngroup, self.scores))
    def save_outcomes(self, outcomes):
        self.empiricalData[-1].save_outcomes(outcomes)
    def need_refresh(self, m, n):
        scoreSequence = self.empiricalData[-1]
        if len(scoreSequence.data) < 20:
            return True


def get_population_fractions(m, n):
    return (m + 1.) / n, float(n - m) / n

def choose_empirical_vs_predicted_mean(scoreSequence, strategyScorer, m, n):
    meanPred, lowerPred, upperPred = strategyScorer.predict_delta(m, n)
    t = scoreSequence.fit_stationary(m, n)
    if not t: # no empirical prediction
        return meanPred
    meanEmp, lowerEmp, upperEmp = t
    strategyScorer.lower = lowerEmp # assign empirical bounds
    strategyScorer.upper = upperEmp
    if lowerPred <= meanEmp and meanEmp <= upperPred:
        return meanEmp
    elif lowerEmp <= meanPred and meanPred <= upperEmp:
        return meanPred
    elif lowerEmp <= upperPred and upperPred <= upperEmp:
        return (lowerEmp + upperPred) / 2.
    elif lowerPred <= upperEmp and upperEmp <= upperPred:
        return (lowerPred + upperEmp) / 2.
    else:
        return meanEmp

class ScoreSequence(object):
    'exponential fit to sequence of avg. scores'
    def __init__(self, n, scores):
        self.n = n
        self.scoreDict = dict(CC=scores[0], CD=scores[1], DC=scores[2], 
                              DD=scores[3])
        self.scoreDict2 = dict(CC=scores[0], DC=scores[1], CD=scores[2], 
                              DD=scores[3])
        self.min = min(scores)
        self.max = max(scores)
        self.data = []
    def save_outcomes(self, outcomes):
        if not outcomes: # no data, so nothing to save
            return
        self.data.append((sum([self.scoreDict[o] for o in outcomes]) 
                          / float(len(outcomes)),
                          sum([self.scoreDict2[o] for o in outcomes]) 
                          / float(len(outcomes))))
    def fit_stationary(self, m, n):
        if len(self.data) < 3: # sequence too short to fit
            return None
        def func(x, a, b, c):
            return a * numpy.exp(-b * x) + c
        xdata = numpy.arange(len(self.data))
        fA, fB = get_population_fractions(m, n)
        ydata = numpy.array([(fB * t[0] - fA * t[1]) for t in self.data])
        try:
            popt, pcov = optimize.curve_fit(func, xdata, ydata,
                                            (1., 1., ydata[-1]))
            try:
                stddev = sqrt(pcov[2,2])
            except (TypeError,ValueError):
                stddev = None
        except RuntimeError: # handle instant convergence case...
            try:
                popt, pcov = optimize.curve_fit(func, xdata, ydata, 
                                                (1., 20., ydata[-1]))
            except RuntimeError: # unable to fit, so give up!
                return None
            diff = numpy.array(self.data[1:]) - popt[2]
            stddev = sqrt((diff * diff).mean() / len(diff))
        y = func(xdata, *popt) # compute deviation from our curve fit
        diff = y - ydata
        sigma = sqrt((diff * diff).mean() / len(diff))
        if stddev is None:
            #print '\trefit:', ydata, diff
            if len(self.data) < 4: # sequence too short to fit
                #print '\tCATCH'
                stddev = self.max - self.min # sigma not accurate
            else:
                stddev = sigma
        mean = popt[2]
        lower = mean - 2. * stddev # 95% confidence interval
        upper = mean + 2. * stddev
        # try to improve a bound based on last point fit
        if popt[0] > 0:
            upper = tightest_bound(upper, y[-1], 2 * sigma)
        else:
            lower = tightest_bound(lower, y[-1], -2 * sigma)
        mean = restrict_range(mean, lower, upper)
        mean = restrict_range(mean, self.min, self.max)
        #print '\tempirical %1.3f %1.3f %1.3f' % (mean, lower, upper)
        self.mean, self.lower, self.upper = mean, lower, upper
        return mean, lower, upper

def tightest_bound(bound1, last, diff):
    bound2 = last + diff
    if bound1 * diff < bound2 * diff:
        return bound1
    else:
        return bound2

def restrict_range(x, bottom, top):
    'restrict x to interval [bottom, top]'
    if x > top:
        return top
    elif x < bottom:
        return bottom
    return x

def add_noise(p, epsilon):
    return p * (1. - epsilon) + (1. - p) * epsilon

def add_noise_vector(l, epsilon):
    return [add_noise(p, epsilon) for p in l]

if __name__ == '__main__':
    import sys
    args = [int(s) for s in sys.argv[1:4]] + sys.argv[4:]
    save_tournaments(*args)
