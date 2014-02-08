#!/usr/bin/python
#$ -S /usr/bin/python
#$ -cwd
#$ -q medium
#$ -o output
#$ -e errors

import os, sys, csv
sys.path.append(os.getcwd())
import fig
import players

def run_score_diffs(player, minSample=20, epsilon=0.05, n=100, runName=None):
    minSample, epsilon, n = int(minSample), float(epsilon), int(n)
    if not runName:
        runName = '%s_%s_score_diffs' %(player, str(epsilon))
    pvec = getattr(players, player)
    diffs, counts = fig.collect_score_diffs(pvec, minSample, 
                                            n=n, epsilon=epsilon)
    with open(runName + '.csv', 'wb') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(diffs)
        csvwriter.writerow(counts)


if __name__ == '__main__':
    if len(sys.argv[1:]) >= 2:
        run_score_diffs(*sys.argv[1:])
    else:
        print '''%s player [minSample, default 20] [epsilon, def 0.05]
    [n, def 100] [runName]
''' % sys.argv[0]

