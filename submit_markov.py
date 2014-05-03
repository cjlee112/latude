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

def run_tournaments(player, player2, nIp, epsilon=0.05, n=100,
                    nrun=1000, ncpu=10, runName=None):
    nIp, epsilon, n, nrun, ncpu = int(nIp), float(epsilon), int(n), \
        int(nrun), int(ncpu)
    if not runName:
        runName = '%s_%s_%s_%s_%s' %(player, player2, str(nIp), str(n), 
                                     str(epsilon))
    playerArgs = getattr(players, player)
    try:
        groupArgs = dict(klass=playerArgs['klass'])
        playerArgs = playerArgs.copy()
        del playerArgs['klass']
    except KeyError:
        groupArgs = {}
    playerArgs2 = getattr(players, player2)

    def get_player(nothers, scores, eps):
        return info.GroupPlayer(nothers, scores, strategyKwargs=playerArgs,
                                name=player, **groupArgs)

    info.save_tournaments(nIp, n, nrun - ncpu + 1, 
                          runName + ".log", epsilon=epsilon, 
                          klass=get_player, strategyKwargs=playerArgs2, 
                          name=player2, 
                          selectionFunction=info.exp_imitation,
                          tournamentClass=info.MultiplayerTournament2,
                          scores=(2, -1, 3, 0))


if __name__ == '__main__':
    if len(sys.argv[1:]) >= 2:
        run_tournaments(*sys.argv[1:])
    else:
        print '''%s player player2 nplayer1 [epsilon, default 0.05] [n, def 100]
    [nrun, def 1000] [ncpu def 10] [runName] 

RUNS: Markov player vs. Markov player tournaments
''' % sys.argv[0]
