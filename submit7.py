import subprocess
import sys

def submit_job(chi, kappa, epsilon=0.05, nIp=1, n=100,
               nrun=1000, ncpu=20, queue='medium', runName=None, player='zd'):
    if not runName:
        runName = '%s_%s_%s_%s' %(player, str(chi), str(kappa), str(epsilon))
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
import players

pvec = players.ss_vector(%(chi)s, kappa=%(kappa)s)

info.save_tournaments(%(nIp)s, %(n)s, %(nrun)s - %(ncpu)s + 1, 
                      "%(runName)s.log", epsilon=%(epsilon)s, 
                      klass=info.InferGroupPlayer2, pvec=pvec,
                      name="%(player)s", selectionFunction=info.exp_imitation,
                      tournamentClass=info.MultiplayerTournament2,
                      scores=(2, -1, 3, 0))
''' % dict(player=player, nIp=str(nIp), n=str(n), runName=runName,
           epsilon=str(epsilon), queue=queue, ncpu=str(ncpu), nrun=str(nrun),
           chi=chi, kappa=kappa)
    script = runName + '.py'
    with open(script, 'w') as ifile:
        ifile.write(s)
    subprocess.call(['qsub', script])


if __name__ == '__main__':
    if len(sys.argv[1:]) >= 2:
        submit_job(*sys.argv[1:])
    else:
        print '''%s chi kappa [epsilon, default 0.05] [nIp, def 1] [n, def 100]
    [runName] [nrun, def 1000] [ncpu def 20] [queue, def medium]
''' % sys.argv[0]
