import subprocess
import sys

def submit_job(player, player2, nIp, epsilon=0.05, n=100,
               nrun=1000, ncpu=10, queue='medium', runName=None):
    if not runName:
        runName = '%s_%s_%s_%s_%s' %(player, player2, str(nIp), str(n), 
                                     str(epsilon))
    s = '''#!/usr/bin/python
#$ -S /usr/bin/python
#$ -cwd
#$ -t 1-%(ncpu)s:1
#$ -q %(queue)s
#$ -l mem=1G
#$ -o output
#$ -e errors

import os, sys
sys.path.append(os.getcwd())
import info
from players import *

def get_player(nothers, scores, eps):
    return info.GroupPlayer(nothers, scores, 
                      strategyKwargs=dict(pvec=%(player)s),
                      name="%(player)s")

info.save_tournaments(%(nIp)s, %(n)s, %(nrun)s - %(ncpu)s + 1, 
                      "%(runName)s.log", epsilon=%(epsilon)s, 
                      klass=get_player, pvec=%(player2)s, name="%(player2)s", 
                      selectionFunction=info.exp_imitation,
                      tournamentClass=info.MultiplayerTournament2)
''' % dict(player=player, nIp=str(nIp), n=str(n), runName=runName,
           epsilon=str(epsilon), queue=queue, ncpu=str(ncpu), nrun=str(nrun),
           player2=player2)
    script = runName + '.py'
    with open(script, 'w') as ifile:
        ifile.write(s)
    subprocess.call(['qsub', script])


if __name__ == '__main__':
    if len(sys.argv[1:]) >= 2:
        submit_job(*sys.argv[1:])
    else:
        print '''%s player player2 nplayer1 [epsilon, default 0.05] [n, def 100]
    [runName] [nrun, def 1000] [ncpu def 10] [queue, def medium]
''' % sys.argv[0]
