#!/usr/bin/python
#$ -S /usr/bin/python
#$ -cwd
#$ -q medium
#$ -o output
#$ -e errors

import os, sys
sys.path.append(os.getcwd())
import info
import players

def run_tournaments(player, nIp, epsilon=0.05, n=100,
                    nrun=1000, ncpu=20, runName=None):
    nIp, epsilon, n, nrun, ncpu = int(nIp), float(epsilon), int(n), \
        int(nrun), int(ncpu)
    if not runName:
        runName = '%s_%s_%s_%s' %(player, str(nIp), str(n), str(epsilon))
    opponent = getattr(players, player)
    if epsilon == 0.:
        klass = info.InferGroupPlayerZeroNoise
    else:
        klass=info.InferGroupPlayer2

    info.save_tournaments(nIp, n, nrun - ncpu + 1, 
                          runName + ".log", epsilon=epsilon, 
                          klass=klass, strategyKwargs=opponent,
                          name=player, selectionFunction=info.exp_imitation,
                          tournamentClass=info.MultiplayerTournament2,
                          scores=(2, -1, 3, 0))


if __name__ == '__main__':
    if len(sys.argv[1:]) >= 2:
        run_tournaments(*sys.argv[1:])
    else:
        print '''%s player nIp [epsilon, default 0.05] [n, def 100]
    [nrun, def 1000] [ncpu def 20] [runName]

RUNS: IP0 vs. Markov player tournaments
''' % sys.argv[0]
