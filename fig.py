import info
import players
from matplotlib import pyplot
import numpy
import random
import csv
import info_test

def plot_selfscore(kappa=None, xvals=numpy.arange(0., 1.001, 0.01),
                   scores=(2, -1, 3, 0), epsilon=0.05, **kwargs):
    yvals = []
    for chi in xvals:
        pvec = players.ss_vector(chi, kappa=kappa, **kwargs)
        pvec = info.add_noise_vector(pvec, epsilon)
        yvals.append(info.stationary_scores(pvec, pvec, scores)[0])
    pyplot.plot(xvals, yvals)

def selfscore_fig():
    plot_selfscore(epsilon=0.001)
    plot_selfscore(0., epsilon=0.001)
    plot_selfscore(epsilon=0.01)
    plot_selfscore(0., epsilon=0.01)
    plot_selfscore(epsilon=0.1)
    plot_selfscore(0., epsilon=0.1)
    pyplot.ylim(ymin=-.1, ymax=2.1)
    pyplot.xlabel('$\chi$')
    pyplot.ylabel('stationary score')
    pyplot.tight_layout()


def phi_chi_fig():
    plot_selfscore(phi=0.01)
    plot_selfscore(0., phi=0.01)
    plot_selfscore(phi=0.1)
    plot_selfscore(0., phi=0.1)
    plot_selfscore(phi=0.2)
    plot_selfscore(0., phi=0.2)
    pyplot.ylim(ymin=-.1, ymax=2.1)
    pyplot.xlabel('chi')
    pyplot.ylabel('stationary score')
    pyplot.show()


def plot_zd_pvec(xvals=numpy.arange(-.33, 1.001, 0.01), 
                   epsilon=0., **kwargs):
    yvals = []
    for chi in xvals:
        pvec = players.ss_vector(chi, **kwargs)
        pvec = info.add_noise_vector(pvec, epsilon)
        yvals.append(pvec)
    pyplot.plot(xvals, [t[0] for t in yvals])
    pyplot.plot(xvals, [t[1] for t in yvals])
    pyplot.plot(xvals, [t[2] for t in yvals])
    pyplot.plot(xvals, [t[3] for t in yvals])


def plot_zd_check(chi=-.33, n=100, scores=(2, -1, 3, 0), **kwargs):
    if chi is not None:
        zdvec = players.ss_vector(chi, **kwargs)
    else:
        zdvec = [random.random() for j in range(4)]
    xvals = []
    yvals = []
    for i in range(n):
        pvec = [random.random() for j in range(4)]
        t = info.stationary_scores(zdvec, pvec, scores)
        xvals.append(t[0])
        yvals.append(t[1])
    pyplot.plot(xvals, yvals, '.')

def plot_popscore_diff(pvec, hisProbs=(1., 0., 1., 0.), n=100, 
                       scores=(2, -1, 3, 0), 
                       epsilon=0.05, selfCooperate=True):
    myProbs = info.add_noise_vector(pvec, epsilon)
    hisProbs = info.add_noise_vector(hisProbs, epsilon)
    if selfCooperate:
        sAA, sBB = info.self_scores(hisProbs, scores, epsilon)
    else:
        sAA = info.stationary_scores(myProbs, myProbs, scores)[0]
        sBB = info.stationary_scores(hisProbs, hisProbs, scores)[0]
    l = []
    for m in range(1, n):
        myBase = sAA * float(m) / n
        hisFrac = float(n - m) / n
        hisBase = sBB * float(n - m - 1) / n
        myFrac = float(m + 1) / n
        l.append(myBase - hisBase + 
                 info.population_score_diff(myFrac, hisFrac, myProbs, hisProbs,
                                            scores))
    pyplot.plot(range(1,n), l)
    pyplot.xlabel('population fraction (%)')
    pyplot.ylabel('fitness difference')

def plot_popscore_diff2(pvec, hisProbs=players.tft, n=100, 
                        scores=(2, -1, 3, 0), epsilon=0.05, pvecSelf=None,
                        infPlayer=False, label=None, labelX=1, **kwargs):
    if infPlayer:
        pvecSelf = players.allc
    elif pvecSelf is None:
        pvecSelf = pvec
    myProbs = info.add_noise_vector(pvecSelf, epsilon)
    hisProbs = info.add_noise_vector(hisProbs, epsilon)
    sAA = info.stationary_scores(myProbs, myProbs, scores)[0]
    sBB = info.stationary_scores(hisProbs, hisProbs, scores)[0]
    l = []
    for m in range(1, n):
        if infPlayer:
            pvec = info.diff_optimum(m, n, hisProbs, scores, epsilon=epsilon)[0]
        myProbs = info.add_noise_vector(pvec, epsilon)
        myBase = sAA * float(m) / n
        hisFrac = float(n - m) / n
        hisBase = sBB * float(n - m - 1) / n
        myFrac = float(m + 1) / n
        l.append(myBase - hisBase + 
                 info.population_score_diff(myFrac, hisFrac, myProbs, hisProbs,
                                            scores))
    pyplot.plot(range(1,n), l, **kwargs)
    if label:
        pyplot.text(labelX, l[labelX - 1], label)
    return l

def popscore_fig(**kwargs):
    plot_popscore_diff2(0, infPlayer=True, label='INFO', labelX=90, linewidth=3,
                        **kwargs)
    plot_popscore_diff2(players.allc, label='ALLC', **kwargs)
    plot_popscore_diff2(players.alld, label='ALLD', labelX=50, **kwargs)
    plot_popscore_diff2(players.zdr2, label='ZDR0.5', labelX=50, **kwargs)
    plot_popscore_diff2(players.zdr2, pvecSelf=players.allc, linestyle='--',
                        label='ZDt', labelX=90, **kwargs)
    plot_popscore_diff2(players.alld, pvecSelf=players.allc, linestyle=':',
                        label='ConDef', labelX=20, **kwargs)
    pyplot.xlabel('population fraction (%)')
    pyplot.ylabel('fitness difference')
    pyplot.tight_layout()

def zdr_popscore_fig(hisProbs=players.zdr2, epsilon=0., **kwargs):
    popscore_fig(hisProbs=hisProbs, epsilon=epsilon, **kwargs)
        
def inf_popscore_fig():
    plot_popscore_diff2(0, infPlayer=True, epsilon=0)
    plot_popscore_diff2(0, infPlayer=True, epsilon=0.01)
    plot_popscore_diff2(0, infPlayer=True, epsilon=0.05)
    plot_popscore_diff2(0, infPlayer=True, epsilon=0.1)
    pyplot.xlabel('population fraction (%)')
    pyplot.ylabel('fitness difference')
    pyplot.show()

def make_popscore_figs(epsilons=(0, 0.05), **kwargs):
    'produce a set of popscore figs for strategies specified by kwargs'
    for name, strategy in kwargs.items():
        for epsilon in epsilons:
            fname = '%s_%s.eps' % (name, str(epsilon))
            print 'generating %s...' % fname
            pyplot.figure()
            popscore_fig(hisProbs=strategy, epsilon=epsilon)
            pyplot.savefig(fname)

def default_popscore_figs():
    'make standard popscore figs for art-of-war paper'
    make_popscore_figs(allc=players.allc, alld=players.alld, 
                       tft=players.tft, wsls=players.wsls, 
                       zdr2=players.zdr2, zdx=players.zdx)
        
def read_csv(csvfile):
    with open(csvfile, 'rb') as ifile:
        return [t[:2] + [int(i) for i in t[2:6]] + [float(x) for x in t[6:]]
                for t in csv.reader(ifile)]

def plot_zd_selection(data, filterFunc, label=None):
    l = [(t[-2], (float(t[4]) / t[5]) / (float(t[2]) / t[3])) for t in data
         if filterFunc(t)]
    l.sort()
    pyplot.semilogy([t[0] for t in l], [t[1] for t in l], marker='o')
    if label:
        print 'label', l[0][0], l[0][1], label
        pyplot.text(l[0][0], l[0][1], label)

def plot_zd_data(data):
    plot_zd_selection(data, lambda t:t[-1]==2 and t[-3]==0.01 and t[-2]>=0, 
                      '$\epsilon=0.01$')
    plot_zd_selection(data, lambda t:t[-1]==2 and t[-3]==0.05 and t[-2]>=0, 
                      '$\epsilon=0.05$')
    plot_zd_selection(data, lambda t:t[-1]==2 and t[-3]==0.1 and t[-2]>=0, 
                      '$\epsilon=0.1$')
    plot_zd_selection(data, lambda t:t[-1]==0 and t[-3]==0.05 and t[-2]>=0, 
                      '$\kappa=0,\epsilon=0.05$')
    pyplot.xlim(xmin=0, xmax=1)
    pyplot.xlabel('$\chi$')
    pyplot.ylabel('robustness')
    pyplot.tight_layout()

def zd_robustness_fig(csvfile='zd_results.csv'):
    data = read_csv(csvfile)
    plot_zd_data(data)
    
def plot_identify_cdf(pvec, n=10000, label='', **kwargs):
    l = [info.time_to_identify(pvec,  **kwargs) for i in range(n)]
    l.sort()
    pyplot.plot(l, numpy.arange(0., 1., 1./n), label=label)

def time_to_identify_fig(pvecs=(players.tft, players.allc, players.alld,
                                players.wsls, players.zdr2, players.zdx),
                         labels=('TFT', 'ALLC', 'ALLD', 'WSLS', 'ZDR0.5', 
                                 'ZDX'),
                         **kwargs):
    for i,pvec in enumerate(pvecs):
        plot_identify_cdf(pvec, label=labels[i], **kwargs)
    pyplot.xlabel('rounds')
    pyplot.ylabel('cumulative probability')
    pyplot.legend()
    pyplot.tight_layout()


def selfscore_vs_zdr(pvec=players.zdr2, n=1000, scores=(2,-1,3,0)):
    xvals = []
    yvals = []
    for i in range(n):
        pvec2 = [random.random() for j in range(4)]
        selfscore = info.stationary_scores(pvec2, pvec2, scores)[0]
        zdrscore = info.stationary_scores(pvec, pvec2, scores)[0]
        xvals.append(zdrscore)
        yvals.append(selfscore)
    return xvals, yvals

def zdx_neutral(n, m, chi, s_RR):
    return s_RR * (n - 1. - m) / (n - m * (chi + 1.))
def zdr_neutral(n, m, chi, s_RR, kappa=2.):
    return (s_RR * (n - 1. - m) - kappa * (m * chi - 1.)) / (n - m * (chi + 1.))

def plot_zd_neutrality(pvec, n=100, m=1, chiMin=-1./3, chiMax=1.001, step=0.01, 
                       scores=(2,-1,3,0)):
    s_RR = info.stationary_scores(pvec, pvec, scores)[0]
    chivals = numpy.arange(chiMin, chiMax, step)
    zdxvals = []
    zdrvals = []
    for chi in chivals:
        zdx = players.ss_vector(chi, kappa=0.)
        s_IR = info.stationary_scores(zdx, pvec, scores)[0]
        zdxvals.append(s_IR - zdx_neutral(n, m, chi, s_RR))
        zdr = players.ss_vector(chi)
        s_IR = info.stationary_scores(zdr, pvec, scores)[0]
        zdrvals.append(s_IR - zdr_neutral(n, m, chi, s_RR))
    pyplot.plot(chivals, zdxvals)
    pyplot.plot(chivals, zdrvals)

def beta_stddev(m, n):
    'compute std deviation of beta posterior for each m value'
    a = m + 1.
    b = (n - m + 1.)
    return numpy.sqrt((a * b) / ((n + 2.) * (n + 2.) * (n + 3.)))

def plot_selection_stddev(n=1000., neutral=0.01):
    mvals = numpy.arange(n)
    svals = mvals / n
    stddevs = beta_stddev(mvals, n)
    pyplot.semilogx(svals / neutral, stddevs / svals)
    pyplot.xlabel('selection relative to neutral')
    pyplot.ylabel('estimation error')

        
def sample_id_odds(nsample=1000, func=info_test.recognize_other, 
                   **kwargs):
    odds = []
    pvals = []
    for i in range(nsample):
        t = func(**kwargs)
        odds.append(t[0])
        pvals.append(t[1])
    odds.sort()
    pvals.sort()
    return odds, pvals

def plot_roc(pvals1, pvals2):
    j = 0
    n = len(pvals1)
    y = numpy.zeros(n)
    for i,p in enumerate(pvals1):
        while j < len(pvals2) and pvals2[j] <= p:
            j += 1
        y[i] = j
    pyplot.plot(numpy.arange(n) / float(n), y / float(len(pvals2)))
