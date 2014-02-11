import glob, os.path
from scipy import stats
import csv

def get_log_params(logname):
    'return params dict from standard logfile name'
    logname = os.path.basename(logname)
    l = logname[:logname.rindex('.')].split('_')
    if len(l) == 5:
        return dict(invader=l[0], defender=l[1], 
                    start=int(l[2]), n=int(l[3]), epsilon=float(l[4]))
    elif len(l) == 4:
        start=int(l[1])
        n=int(l[2])
        if start > n / 2:
            return dict(invader=l[0], symbol='M', defender='I', 
                        start=n - start, n=n, epsilon=float(l[3]))
        else:
            return dict(invader='I', defender=l[0], 
                        start=start, n=n, epsilon=float(l[3]))

def analyze_log(filename, invader, start, n):
    'extract wins, runs, pUnder and pOver from logfile'
    pNeutral = float(start) / n
    wins = runs = 0
    with open(filename, 'rU') as ifile:
        for line in ifile:
            l = line.strip().split()
            if len(l) == 2:
                runs += 1
                if l[0].isdigit() and not invader.isdigit():
                    if invader == 'I':
                        invader = '1'
                    else:
                        invader = '0'
                if l[0] == invader:
                    wins += 1
    b = stats.binom(runs, pNeutral)
    return wins, runs, b.sf(wins - 1), b.cdf(wins)

def analyze_logs(pattern='*.log', paramsFunc=get_log_params):
    'analyze log files matching pattern, with params extracted from filenames'
    l = []
    for logfile in glob.glob(pattern):
        d = paramsFunc(logfile)
        if d:
            wins, runs, pOver, pUnder = analyze_log(logfile, 
                                            d.get('symbol', d['invader']),
                                            d['start'], d['n'])
            d['wins'] = wins
            d['runs'] = runs
            d['pOver'] = pOver
            d['pUnder'] = pUnder
            l.append(d)
    return l

columns = ('invader', 'defender', 'start', 'n', 'wins', 'runs',
           'pUnder', 'pOver', 'epsilon')

def save_csv(results, filename='results.csv', cols=columns):
    'write specified columns to csv file from list of dictionaries'
    with open(filename, 'wb') as csvfile:
        writer = csv.writer(csvfile)
        for d in results:
            writer.writerow([d[k] for k in cols])


def assess_zd_chi(filename='zd_results.csv', pattern='zd_*.log', start=1, 
                  n=100):
    'generate csv results from zd_CHI_KAPPA_EPSILON.log files'
    def get_zd_params(logname):
        logname = os.path.basename(logname)
        l = logname[:logname.rindex('.')].split('_')
        return dict(invader='I', defender=l[0], start=start, n=n, 
                    chi=float(l[1]), kappa=float(l[2]), epsilon=float(l[3]))
    data = analyze_logs(pattern, get_zd_params)
    save_csv(data, filename, columns + ('chi', 'kappa'))

if __name__ == '__main__':
    l = analyze_logs()
    save_csv(l)
