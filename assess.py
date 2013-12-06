import glob
from scipy import stats
import csv

def get_log_params(logname):
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

def analyze_logs(pattern='*.log'):
    l = []
    for logfile in glob.glob(pattern):
        d = get_log_params(logfile)
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

def save_csv(results, filename='results.csv'):
    with open(filename, 'wb') as csvfile:
        writer = csv.writer(csvfile)
        for d in results:
            writer.writerow((d['invader'], d['defender'], d['start'],
                             d['n'], d['wins'], d['runs'],
                             d['pUnder'], d['pOver'], d['epsilon']))

if __name__ == '__main__':
    l = analyze_logs()
    save_csv(l)
