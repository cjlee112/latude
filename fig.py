import info
import players
try:
    from matplotlib import pyplot
except ImportError:
    pass
import numpy
import random
import csv
import info_test
import glob
import os.path

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

def popscore_fig(ip0Scores=None, ip0X=90, allcX=1, alldX=50, zdr2X=50,
                 zdtX=90, condefX=20, **kwargs):
    if ip0Scores is not None: # plot empirical score curve
        n = len(ip0Scores) + 1
        pyplot.plot(range(1, n), ip0Scores, linewidth=3)
        pyplot.text(90, ip0Scores[ip0X - 1], 'IP0')
    #plot_popscore_diff2(0, infPlayer=True, label='INFO', labelX=90, linewidth=3,
    #                    **kwargs)
    plot_popscore_diff2(players.allc, label='ALLC', labelX=allcX, **kwargs)
    plot_popscore_diff2(players.alld, label='ALLD', labelX=alldX, **kwargs)
    plot_popscore_diff2(players.zdr2, label='ZDR0.5', labelX=zdr2X, **kwargs)
    plot_popscore_diff2(players.zdr2, pvecSelf=players.allc, linestyle='--',
                        label='ZDt', labelX=zdtX, **kwargs)
    plot_popscore_diff2(players.alld, pvecSelf=players.allc, linestyle=':',
                        label='ConDef', labelX=condefX, color='k', 
                        linewidth=2, **kwargs)
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

def plot_zd_selection(data, filterFunc=lambda t:True, label=None, dy=0, 
                      marker='o', **kwargs):
    l = [(t[-2], (float(t[4]) / t[5]) / (float(t[2]) / t[3])) for t in data
         if filterFunc(t)]
    l.sort()
    pyplot.semilogy([t[0] for t in l], [t[1] for t in l], marker=marker, 
                    **kwargs)
    if label:
        pyplot.text(l[0][0], l[0][1] + dy, label)

def plot_zd_data(data):
    plot_zd_selection(data, lambda t:t[-1]==2 and t[-3]==0. and t[-2]>=0, 
                      '$\epsilon=0$')
    plot_zd_selection(data, lambda t:t[-1]==2 and t[-3]==0.01 and t[-2]>=0, 
                      '$\epsilon=0.01$', -0.3)
    plot_zd_selection(data, lambda t:t[-1]==2 and t[-3]==0.05 and t[-2]>=0, 
                      '$\epsilon=0.05$', -1)
    plot_zd_selection(data, lambda t:t[-1]==2 and t[-3]==0.1 and t[-2]>=0, 
                      '$\epsilon=0.1$', 1)
    plot_zd_selection(data, lambda t:t[-1]==0 and t[-3]==0.05 and t[-2]>=0, 
                      '$\kappa=0,\epsilon=0.05$')
    pyplot.xlim(xmin=0, xmax=1)
    pyplot.xlabel('$\chi$')
    pyplot.ylabel('IP0 invasion success')
    pyplot.tight_layout()

def zd_robustness_fig(csvfile='zd_results.csv', tagfile='tag_results.csv'):
    data = read_csv(csvfile)
    plot_zd_data(data)
    data = read_csv(tagfile)
    plot_zd_selection(data, label='ConSwitch ($\epsilon=0$)', marker='^', 
                      linestyle='--', color='k')
    
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
    'get sorted sample of odds and pvalues from specified ID challenge func'
    odds = []
    pvals = []
    for i in range(nsample):
        t = func(**kwargs)
        odds.append(t[0])
        pvals.append(t[1])
    odds.sort()
    pvals.sort()
    return odds, pvals

def id_odds_dist(ncycle, pvec=None, epsilon=0.05, **kwargs):
    'get sorted odds and pvalues for self vs. pvec identification sample'
    oddsIP, pvalsIP = sample_id_odds(ncycle=ncycle, epsilon=epsilon,
                                     func=info_test.recognize_self, **kwargs)
    oddsGP, pvalsGP = sample_id_odds(ncycle=ncycle, pvec=pvec, epsilon=epsilon,
                                     func=info_test.recognize_other, **kwargs)
    return oddsIP, pvalsIP, oddsGP, pvalsGP

def calc_roc(pvalsFP, pvalsTP):
    'compute the ROC curve for the specified FP vs. TP score distributions'
    l = [(p, 0) for p in pvalsFP] + [(p, 1) for p in pvalsTP]
    l.sort()
    nTP = float(len(pvalsTP))
    nFP = float(len(pvalsFP))
    c = [0, 0]
    i = 0
    roc = [(0., 0.)]
    while i < len(l):
        cut = l[i][0]
        while i < len(l) and l[i][0] <= cut:
            c[l[i][1]] += 1
            i += 1
        roc.append((c[0] / nFP, c[1] / nTP))
    return roc

def calc_auc(pvalsFP, pvalsTP):
    'compute the ROC area-under-the-curve'
    roc = calc_roc(pvalsFP, pvalsTP)
    xlast, ylast = roc[0]
    auc = 0.
    for x,y in roc[1:]: # numerically integrate area under curve
        auc += (x - xlast) * (y + ylast) / 2.
        xlast, ylast = x, y
    return auc

def calc_id_auc(ncycle, pvec=None, epsilon=0.05, **kwargs):
    'compute AUC for self vs. pvec identification'
    oddsIP, pvalsIP, oddsGP, pvalsGP = id_odds_dist(ncycle, pvec, epsilon, 
                                                    **kwargs)
    return calc_auc(pvalsIP, pvalsGP)
    
def plot_auc(rounds=range(2, 11), **kwargs):
    'plot AUC over specified number of infogain cycles'
    l = []
    for ncycle in rounds:
        l.append(calc_id_auc(ncycle, **kwargs))
    pyplot.plot(rounds, l)

def id_auc_fig(pvec=players.zdr2, epsilons=(0.1, 0.05, 0.01, 0.)):
    'AUC figure for specified pvec, for different epsilon values'
    for epsilon in epsilons:
        plot_auc(pvec=pvec, epsilon=epsilon)
    pyplot.ylim(ymax=1.01)
    pyplot.xlabel('number of infogain rounds')
    pyplot.ylabel('AUC')
    pyplot.tight_layout()


def plot_roc(ncycle=10, pvec=None, epsilon=0.05, label=None, **kwargs):
    'plot ROC for specified number of infogain cycles'
    oddsIP, pvalsIP, oddsGP, pvalsGP = id_odds_dist(ncycle, pvec, epsilon, 
                                                    **kwargs)
    roc = calc_roc(pvalsIP, pvalsGP)
    pyplot.plot([t[0] for t in roc], [t[1] for t in roc], label=label)

def id_roc_fig(strategies=(players.zdr2, players.zdx, players.tft,
                           players.wsls, players.allc, players.alld),
               labels=('ZDR2', 'ZDX', 'TFT', 'WSLS', 'ALLC', 'ALLD'),
               epsilon=0.05):
    'ROC figure for identification of specified list of strategies'
    for i,pvec in enumerate(strategies):
        plot_roc(pvec=pvec, epsilon=epsilon, label=labels[i])
    pyplot.xlim(xmin=-0.01, xmax=0.4)
    pyplot.xticks((0., 0.1, 0.2, 0.3, 0.4))
    pyplot.ylim(ymin=0.8, ymax=1.01)
    pyplot.xlabel('FP')
    pyplot.ylabel('TP')
    pyplot.legend(loc='lower right')
    pyplot.tight_layout()


def collect_score_diffs(pvec=None, minSample=100, iSample=-1, 
                        nIp=1, n=100, epsilon=0.05, ncycle=10000, nrun=50):
    'draw sample of S_I - S_G as a function of m'
    diffs = numpy.zeros(n)
    counts = numpy.zeros(n)
    i = 0
    while counts[iSample] < minSample or i < nrun:
        l, tour = info.check_accuracy(nIp, n, pvec, ncycle, epsilon=epsilon)
        i += 1
        for d in l:
            m = d['#I']
            diffs[m] += d['I'] - d['M'] # difference in average scores
            counts[m] += 1
    return diffs, counts

def read_score_diffs(fname):
    'read csv file consisting of one row of diffs and one row of counts'
    with open(fname, 'rb') as csvfile:
        csvreader = csv.reader(csvfile)
        it = iter(csvreader)
        diffs = numpy.array([float(s) for s in it.next()][1:])
        counts = numpy.array([float(s) for s in it.next()][1:])
    return diffs, counts

def get_score_diff_files(pattern='*.csv'):
    'get dict of files of the form allc_0.05_whatever.csv'
    d = {}
    for fname in glob.glob(pattern):
        try:
            player, epsilon = os.path.basename(fname).split('_')[:2]
        except ValueError:
            continue
        epsilon = float(epsilon)
        diffs, counts = read_score_diffs(fname)
        d[(player, epsilon)] = diffs / counts
    return d


def save_popscore_figs(pattern):
    d = get_score_diff_files(pattern)
    for t,sd in d.items():
        player, epsilon = t
        fname = '%s_%d.eps' % (player, int(100 * epsilon))
        pvec = getattr(players, player)
        pyplot.figure()
        if player == 'wsls':
            alldX = 70
        else:
            alldX = 50
        if player == 'zdx':
            condefX = 10
        else:
            condefX = 20
        print 'Saving', fname
        popscore_fig(sd, hisProbs=pvec, epsilon=epsilon, alldX=alldX,
                     condefX=condefX)
        pyplot.savefig(fname)

