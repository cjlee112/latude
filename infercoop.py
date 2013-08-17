from scipy import stats
import numpy



def iid_binomial(n, mvals, my_m, **kwargs):
    'trivial binomial IID model based on current reputation'
    pCoop =  (mvals + 1.) / (n + 2.) # estimate with pseudocounts
    return dict(pCoop=pCoop)


'''
>>> a = numpy.arange(6)
>>> a
array([0, 1, 2, 3, 4, 5])
>>> infercoop.top_binomial(5, a, 4)
{'pCoop': array([ 0.0877915 ,  0.35116598,  0.68038409,  0.89986283,  0.98216735,
        0.99862826])}
'''


def top_binomial(n, mvals, my_m, **kwargs):
    '''Given vector of m values (number of times player i cooperated,
    out of n total trials), and my m value, compute vector of probabilities
    that player i will cooperate with me.

    Assuming:

    * mvals should be a numpy array
    * each player i cooperates with the TOP (most cooperative) p_i players
    * p_i is inferred from binomial model given m,n counts and uninformative
      prior. '''
    # compute fraction who cooperated more than I
    pRank = (mvals >= my_m).sum() / float(len(mvals))
    d = {}
    pCoop = numpy.zeros(len(mvals))
    for i,m in enumerate(mvals):
        try:
            pCoop[i] = d[m]
        except KeyError:
            rv = stats.beta(m + 1, n - m + 1)
            pCoop[i] = d[m] = rv.sf(pRank)
    return dict(pCoop=pCoop)


# empirical approach
'''
>>> a = numpy.array((4,7,2,9,1,8))
>>> rvals = numpy.arange(6)
>>> infercoop.infer_empirical(a, rvals, 5)
(array([ 0.42857143,  0.57142857,  0.28571429,  0.85714286,  0.14285714,
        0.71428571]), array([2, 3, 1, 5, 0, 4]))
'''


def infer_empirical(n, mvals, myRep, lastround, rvals=None, nround=0, **kwargs):
    '''Ignores mvals, instead sums lastround history (stored in rvals)
    to compute pCoop as PL using pseudocounts.

    * p_i is inferred from binomial model given r,n counts and uninformative
      prior. 
    '''
    if rvals is None:
        rvals = numpy.zeros(len(mvals), int)
    rvals = rvals + lastround # sum history of each player over all rounds
    nround += 1
    pCoop =  (rvals + 1.) / (nround + 2.) # estimate with pseudocounts
    return dict(pCoop=pCoop, rvals=rvals, nround=nround)






###############################################################
# remap previous round to current

def remap_reID(reIDs, d):
    '''remap all arrays in dict d, according to reIDs vector '''
    for k,v in d.items():
        if not isinstance(v, numpy.ndarray):
            continue # not an array, no need to remap
        if isinstance(v[0], int):
            a = numpy.zeros(len(reIDs), int)
        else:
            a = numpy.zeros(len(reIDs))
        for i,reID in enumerate(reIDs):
            a[i] = v[reID]
        d[k] = a


class PLModel(object):
    def __init__(self, models):
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
