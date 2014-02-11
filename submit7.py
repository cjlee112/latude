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

def run_zd_tournaments(chi, kappa, epsilon=0.05, nIp=1, n=100,
                       nrun=1000, ncpu=20, player='zd', runName=None):
    chi, kappa, epsilon, nIp, n, nrun, ncpu = float(chi), float(kappa), \
        float(epsilon), int(nIp), int(n), int(nrun), int(ncpu)
    if not runName:
        runName = '%s_%s_%s_%s' %(player, str(chi), str(kappa), str(epsilon))
    pvec = players.ss_vector(chi, kappa=kappa)
    if epsilon == 0.:
        klass = info.InferGroupPlayerZeroNoise
    else:
        klass=info.InferGroupPlayer2

    info.save_tournaments(nIp, n, nrun - ncpu + 1, 
                          runName + ".log", epsilon=epsilon, 
                          klass=klass, pvec=pvec,
                          name=player, selectionFunction=info.exp_imitation,
                          tournamentClass=info.MultiplayerTournament2,
                          scores=(2, -1, 3, 0))


if __name__ == '__main__':
    if len(sys.argv[1:]) >= 2:
        run_zd_tournaments(*sys.argv[1:])
    else:
        print '''%s chi kappa [epsilon, default 0.05] [nIp, def 1] [n, def 100]
    [nrun, def 1000] [ncpu def 20] [playerName def zd] [runName] 

RUNS: IP0 vs. ZD(chi, kappa) player tournaments
''' % sys.argv[0]
