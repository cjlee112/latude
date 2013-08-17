from scipy import stats
import numpy



def iid_binomial(n, mvals, my_m, **kwargs):
    'trivial binomial IID model based on current reputation'
    pCoop =  (mvals + 1.) / (n + 2.) # estimate with pseudocounts
    return dict(pCoop=pCoop)


def iid_binomial_last(n, mvals, my_m, mvalsLast=None, nLast=0, **kwargs):
    '''same as iid_binomial() except it computes pCoop solely based on
    reputation increase from the previous round.
    This should handle any player who suddenly switches to allC, allD,
    or some other value of theta.'''
    if mvalsLast is not None: # compute rep increase from last round
        mvalsRound = mvals - mvalsLast
        nRound = n - nLast
        condition = numpy.logical_and(mvalsRound >= 0, mvalsRound <= nRound)
        pCoop =  numpy.where(condition, (mvalsRound + 1.) / (nRound + 2.), 0.5)
    else: # treat as first round
        pCoop = (mvals + 1.) / (n + 2.)
    return dict(pCoop=pCoop, mvalsLast=numpy.array(mvals), 
                nLast=n) # save last round info



'''
>>> a = numpy.arange(6)
>>> a
array([0, 1, 2, 3, 4, 5])
>>> infercoop.top_binomial(5, a, 4)
{'pCoop': array([ 0.0877915 ,  0.35116598,  0.68038409,  0.89986283,  0.98216735,
        0.99862826])}
'''


def top_binomial(n, mvals, my_m, pRandom=0.1, **kwargs):
    '''Given vector of m values (number of times player i cooperated,
    out of n total trials), and my m value, compute vector of probabilities
    that player i will cooperate with me.

    Assuming:

    * mvals should be a numpy array
    * each player i cooperates with the TOP (most cooperative) p_i players
    * p_i is inferred from binomial model given m,n counts and uninformative
      prior. 
    * pRandom: probability fraction associated with randomly cooperating
      with anyone, rather than according to TOP model.  Makes this less
      of a zero-one boolean function.'''
    # compute fraction who cooperated more than I
    pRank = (mvals >= my_m).sum() / float(len(mvals))
    d = {}
    pCoop = numpy.zeros(len(mvals))
    for i,m in enumerate(mvals):
        try:
            pCoop[i] = d[m]
        except KeyError:
            rv = stats.beta(m + 1, n - m + 1)
            pCoop[i] = d[m] = rv.sf(pRank) * (1. - pRandom) \
                + pRandom * (1. - pRank)
    return dict(pCoop=pCoop)



def top_binomial_last(n, mvals, my_m, mvalsLast=None, nLast=0, pRandom=0.1,
                      **kwargs):
    '''same as top_binomial() except it computes pCoop solely based on
    reputation increase from the previous round'''
    # compute fraction who cooperated more than I
    pRank = (mvals >= my_m).sum() / float(len(mvals))
    d = {}
    pCoop = numpy.zeros(len(mvals))
    if mvalsLast is not None:
        mvalsRound = mvals - mvalsLast
    else:
        mvalsRound = mvals
    nRound = n - nLast
    for i,m in enumerate(mvalsRound):
        try:
            pCoop[i] = d[m]
        except KeyError:
            if m >= 0 and m <= nRound: # make sure remapped results make sense
                rv = stats.beta(m + 1, nRound - m + 1)
                pCoop[i] = d[m] = rv.sf(pRank) * (1. - pRandom) \
                    + pRandom * (1. - pRank)
            else: # something wrong with mapping, so treat as uncertain
                pCoop[i] = 0.5
    return dict(pCoop=pCoop, mvalsLast=numpy.array(mvals), 
                nLast=n) # save last round info


# empirical approach


def iid_empirical(n, mvals, myRep, lastround, rvals=None, nround=0, **kwargs):
    '''Ignores mvals, instead sums lastround history (stored in rvals)
    to compute pCoop as PL using pseudocounts.

    * p_i is inferred from binomial model given r,n counts and uninformative
      prior, over the entire history. 
    '''
    if rvals is None:
        rvals = numpy.zeros(len(mvals), int)
    rvals = rvals + lastround # sum history of each player over all rounds
    nround += 1
    pCoop =  (rvals + 1.) / (nround + 2.) # estimate with pseudocounts
    return dict(pCoop=pCoop, rvals=rvals, nround=nround)


def recent_empirical(n, mvals, myRep, lastround, rvals=None, nround=0, 
                     keepRounds=10, **kwargs):
    """uses last 10 rounds of each player's moves (vs. me) to estimate
    p(coop).  Good for detecting if player switches to different theta
    vs. me than vs. everyone else."""
    if rvals is None: # 1st round so initialize storage of keepRounds rounds
        rvals = numpy.zeros((len(mvals), keepRounds), int)
    else:
        rvals[:,:-1] = rvals[:,1:] # shift history back one step
    rvals[:,-1] = lastround # save last move (vs. me) of each player
    if nround < numpy.shape(rvals)[1]: # count rounds until rvals full
        nround += 1
    pCoop =  (rvals.sum(axis=1) + 1.) / (nround + 2.) # use pseudocounts
    return dict(pCoop=pCoop, rvals=rvals, nround=nround)





###############################################################
# remap previous round to current

def remap_reID(reIDs, d):
    '''remap all arrays in dict d, according to reIDs vector '''
    for k,v in d.items():
        if not isinstance(v, numpy.ndarray):
            continue # not an array, no need to remap
        sh = numpy.shape(v)
        if len(sh) > 1:
            a = numpy.zeros((len(reIDs),) + sh[1:], v.dtype)
        else:
            a = numpy.zeros(len(reIDs), v.dtype)
        for i,reID in enumerate(reIDs):
            a[i] = v[reID]
        d[k] = a


class PLModel(object):
    def __init__(self, models=(iid_binomial, 
                               iid_binomial_last,
                               top_binomial,
                               top_binomial_last,
                               iid_empirical,
                               recent_empirical)):
        self.models = models
        self.data = [{} for m in models] # empty dict = uninformative prior

    def __call__(self, reIDs, n, mvals, myRep, lastround, lastroundMe,
                 pTrans=0.01, **kwargs):
        '''reIDs map current round index --> last round index
        n: total number of games for each player (roughly nRound*(nPlayers-1))
        mvals: reputation of players entering current round (integer 
          representing #times that player chose to cooperate, out of all games)
        lastround: actual move of each player vs. me in last round index
        lastroundMe: actual move we played vs. each player in last round index
        '''
        forward = numpy.ones((len(reIDs), len(self.models))) # uninf. prior
        for i, model in enumerate(self.models):
            data = self.data[i]
            try: # update forward probability based on last round obs
                pCoop = data['pCoop']
                likelihoods = pCoop * lastround + (1. - pCoop) * (1 - lastround)
                data['forward'] = data['pForward'] * likelihoods
                del data['pForward']
            except KeyError:
                pass
            data['lastround'] = lastround # prepare to remap
            data['lastroundMe'] = lastroundMe
            remap_reID(reIDs, data) # remap to current round
            try:
                forward[:,i] = data['forward'] # remapped p(Ot-1,Ht-1=i|O^t-2)
            except KeyError:
                pass
            data.update(kwargs) # pass kwargs to model
            self.data[i] = model(n, mvals, myRep, **data) # run the model

        s = forward.sum(axis=1) # sum over all possible models
        t = s * (pTrans / len(self.models)) # transition probabilities
        s *= (1. + pTrans) # normalization factor
        pCoop = numpy.zeros(len(reIDs))
        for i in range(len(self.models)):
            f = forward[:,i] # forward prob for model i from last round
            f += t # apply transition probabilities 
            f /= s # normalize
            self.data[i]['pForward'] = f # save p(Ht=i|O^t-1)
            pCoop += f * self.data[i]['pCoop'] # compute PL
        return pCoop # PL prediction that each player will cooperate with me
