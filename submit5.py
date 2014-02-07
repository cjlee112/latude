import subprocess
import sys

def submit_job(player, nIp, epsilon=0.05, n=100,
               nrun=1000, ncpu=20, queue='medium', runName=None):
    if not runName:
        runName = '%s_%s_%s_%s' %(player, str(nIp), str(n), str(epsilon))
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

epsilon=%(epsilon)s
if epsilon == 0.:
    klass = info.InferGroupPlayerZeroNoise
else:
    klass=info.InferGroupPlayer2

info.save_tournaments(%(nIp)s, %(n)s, %(nrun)s - %(ncpu)s + 1, 
                      "%(runName)s.log", epsilon=epsilon, 
                      klass=klass, pvec=%(player)s,
                      name="%(player)s", selectionFunction=info.exp_imitation,
                      tournamentClass=info.MultiplayerTournament2,
                      scores=(2, -1, 3, 0))
''' % dict(player=player, nIp=str(nIp), n=str(n), runName=runName,
           epsilon=str(epsilon), queue=queue, ncpu=str(ncpu), nrun=str(nrun))
    script = runName + '.py'
    with open(script, 'w') as ifile:
        ifile.write(s)
    subprocess.call(['qsub', script])


if __name__ == '__main__':
    if len(sys.argv[1:]) >= 2:
        submit_job(*sys.argv[1:])
    else:
        print '''%s player nIp [epsilon, default 0.05] [n, def 100]
    [nrun, def 1000] [ncpu def 20] [queue, def medium] [runName]
''' % sys.argv[0]
