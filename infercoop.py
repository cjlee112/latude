from scipy import stats
import numpy



# empirical approach


def iid_empirical(lastround, rvals=None, nround=0, **kwargs):
    '''Ignores mvals, instead sums lastround history (stored in rvals)
    to compute pCoop as PL using pseudocounts.

    * p_i is inferred from binomial model given r,n counts and uninformative
      prior, over the entire history. 
    '''
    if rvals is None:
        rvals = numpy.zeros(len(lastround), int)
    rvals = rvals + lastround # sum history of each player over all rounds
    nround += 1
    pCoop =  (rvals + 1.) / (nround + 2.) # estimate with pseudocounts
    return dict(pCoop=pCoop, rvals=rvals, nround=nround)


def recent_empirical(lastround, rvals=None, nround=0, 
                     keepRounds=10, **kwargs):
    """uses last 10 rounds of each player's moves (vs. me) to estimate
    p(coop).  Good for detecting if player switches to different theta
    vs. me than vs. everyone else."""
    if rvals is None: # 1st round so initialize storage of keepRounds rounds
        rvals = numpy.zeros((len(lastround), keepRounds), int)
    else:
        rvals[:,:-1] = rvals[:,1:] # shift history back one step
    rvals[:,-1] = lastround # save last move (vs. me) of each player
    if nround < numpy.shape(rvals)[1]: # count rounds until rvals full
        nround += 1
    pCoop =  (rvals.sum(axis=1) + 1.) / (nround + 2.) # use pseudocounts
    return dict(pCoop=pCoop, rvals=rvals, nround=nround)


def tft(lastround, lastroundMe, pGenerous=0.05, **kwargs):
    'generous tit for tat'
    pCoop = numpy.where(lastroundMe, 1., pGenerous)
    return dict(pCoop=pCoop)

def wsls(lastround, lastroundMe, pws=0.99, **kwargs):
    'win-stay-lose-shift'
    pCoop = numpy.where(lastroundMe, pws * lastround, 1. - pws * lastround)
    return dict(pCoop=pCoop)

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
    def __init__(self, models=(iid_empirical,
                               recent_empirical,
                               tft,
                               wsls)):
        self.models = models
        self.data = [{} for m in models] # empty dict = uninformative prior

    def __call__(self, lastround, lastroundMe,
                 pTrans=0.01, **kwargs):
        '''lastround: actual move of each player vs. me in last round index
        lastroundMe: actual move we played vs. each player in last round index
        '''
        forward = numpy.ones((len(lastround), len(self.models))) # uninf. prior
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
            try:
                forward[:,i] = data['forward'] # remapped p(Ot-1,Ht-1=i|O^t-2)
            except KeyError:
                pass
            data.update(kwargs) # pass kwargs to model
            self.data[i] = model(**data) # run the model

        s = forward.sum(axis=1) # sum over all possible models
        t = s * (pTrans / len(self.models)) # transition probabilities
        s *= (1. + pTrans) # normalization factor
        pCoop = numpy.zeros(len(lastround))
        for i in range(len(self.models)):
            f = forward[:,i] # forward prob for model i from last round
            f += t # apply transition probabilities 
            f /= s # normalize
            self.data[i]['pForward'] = f # save p(Ht=i|O^t-1)
            pCoop += f * self.data[i]['pCoop'] # compute PL
        return pCoop # PL prediction that each player will cooperate with me
