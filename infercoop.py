from scipy import stats
import numpy


'''
>>> a = numpy.arange(6)
>>> a
array([0, 1, 2, 3, 4, 5])
>>> infercoop.infer_binomial(4, a, 5)
[0.046656000000000031, 0.23327999999999993, 0.54431999999999992, 0.82079999999999997, 0.95904, 0.99590400000000001]
'''


def infer_binomial(my_m, mvals, n):
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
    l = []
    for m in mvals:
        try:
            l.append(d[m])
        except KeyError:
            rv = stats.beta(m + 1, n - m + 1)
            p = d[m] = rv.sf(pRank)
            l.append(p)
    return l


# empirical approach
'''
>>> a = numpy.array((4,7,2,9,1,8))
>>> rvals = numpy.arange(6)
>>> infercoop.infer_empirical(a, rvals, 5)
(array([ 0.42857143,  0.57142857,  0.28571429,  0.85714286,  0.14285714,
        0.71428571]), array([2, 3, 1, 5, 0, 4]))
'''


def infer_empirical(mvals, rvals, n):
    '''Given vector of m values (number of times player i cooperated,
    out of n total trials), and vector of r values (number of times
    player of rank r cooperated with ME,
    out of n total trials), compute vector of probabilities
    that player i will cooperate with me.

    Assuming:

    * p_i is inferred from binomial model given r,n counts and uninformative
      prior. 

    Returns tuple of (l, ranks), where
    l is vector of cooperation probabilities and
    ranks is the rank (in sorted list of mvals) of each player i'''
    msort = [(t[1], t[0]) for t in enumerate(mvals)]
    msort.sort() # in order of increasing m
    l = numpy.zeros(len(mvals))
    ranks = numpy.zeros(len(mvals), int)
    for r, t in enumerate(msort):
        l[t[1]] = (rvals[r] + 1.) / (n + 2.) # estimate with pseudocounts
        ranks[t[1]] = r
    return l, ranks

