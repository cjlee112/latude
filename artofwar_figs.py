import fig
from matplotlib import pyplot

# data generation for table
# using commit 8911c:
# qsub -t 1-100:1 submit_ip.py zdx2 1 0.05 100 10000 100
# qsub -t 1-100:1 submit_ip.py zdx2 99 0.05 100 10000 100
# qsub -t 1-10:1 submit_markov.py zdx2 allc 1 .05 100 10000 10
# qsub -t 1-10:1 submit_markov.py allc zdx2 1 .05 100 10000 10
# qsub -t 1-10:1 submit_markov.py zdx2 alld 1 .05 100 10000 10
# qsub -t 1-10:1 submit_markov.py alld zdx2 1 .05 100 10000 10
# qsub -t 1-10:1 submit_markov.py zdx2 tft 1 .05 100 10000 10
# qsub -t 1-10:1 submit_markov.py tft zdx2 1 .05 100 10000 10
# qsub -t 1-10:1 submit_markov.py zdx2 wsls 1 .05 100 10000 10
# qsub -t 1-10:1 submit_markov.py wsls zdx2 1 .05 100 10000 10
# qsub -t 1-10:1 submit_markov.py zdx2 zdr2 1 .05 100 10000 10
# qsub -t 1-10:1 submit_markov.py zdr2 zdx2 1 .05 100 10000 10


# data generation for fig 1 - 2 on cassini research/latude
# using commit 80283:
# qsub submit_score_diffs.py allc 100 0
# qsub submit_score_diffs.py allc 100 0.05
# qsub submit_score_diffs.py alld 100 0
# qsub submit_score_diffs.py alld 100 0.05
# qsub submit_score_diffs.py tft 100 0
# qsub submit_score_diffs.py tft 100 0.05
# qsub submit_score_diffs.py wsls 100 0
# qsub submit_score_diffs.py wsls 100 0.05
# qsub submit_score_diffs.py zdr2 100 0
# qsub submit_score_diffs.py zdr2 100 0.05
# using commit 8911c:
# qsub submit_score_diffs.py zdx2 100 0
# qsub submit_score_diffs.py zdx2 100 0.05

def make_popscore_figs(patterns=('80283/[a-w]*.csv', 
                                 '80283/zdr2_*.csv', 
                                 '80283_more/*.csv',
                                 '8911c/zdx2_0*.csv', )):
    'produces fig 1 and 2 of the artofware paper'
    scoreDiffDict = fig.get_score_diff_files(patterns)
    fig.save_popscore_figs(scoreDiffDict)

def make_roc_fig(fname='id_roc.eps'):
    'fig 3a'
    pyplot.figure()
    fig.id_roc_fig()
    print 'Saving', fname
    pyplot.savefig(fname)

def make_auc_fig(fname='id_auc.eps'):
    'fig 3b'
    pyplot.figure()
    fig.id_auc_fig()
    print 'Saving', fname
    pyplot.savefig(fname)

# data generation for fig 4
# qsub -t1-100:1 submit_ip_zd.py 0 2 0.05 1 100 10000 100
# qsub -t1-100:1 submit_ip_zd.py 0.5 2 0.05 1 100 10000 100
# qsub -t1-100:1 submit_ip_zd.py 0.8 2 0.05 1 100 10000 100
# qsub -t1-100:1 submit_ip_zd.py 0.9 2 0.05 1 100 10000 100
# qsub -t1-100:1 submit_ip_zd.py 1 2 0.05 1 100 10000 100
# qsub -t1-100:1 submit_ip_zd.py 0 2 0 1 100 10000 100
# qsub -t1-100:1 submit_ip_zd.py 0.5 2 0 1 100 10000 100
# qsub -t1-100:1 submit_ip_zd.py 0.8 2 0 1 100 10000 100
# qsub -t1-100:1 submit_ip_zd.py 0.9 2 0 1 100 10000 100
# qsub -t1-100:1 submit_ip_zd.py 1 2 0 1 100 10000 100
# qsub -t1-100:1 submit_ip_zd.py 0 2 0.01 1 100 10000 100
# qsub -t1-100:1 submit_ip_zd.py 0.5 2 0.01 1 100 10000 100
# qsub -t1-100:1 submit_ip_zd.py 0.8 2 0.01 1 100 10000 100
# qsub -t1-100:1 submit_ip_zd.py 0.9 2 0.01 1 100 10000 100
# qsub -t1-100:1 submit_ip_zd.py 1 2 0.01 1 100 10000 100
# qsub -t1-100:1 submit_ip_zd.py 0 2 0.1 1 100 10000 100
# qsub -t1-100:1 submit_ip_zd.py 0.5 2 0.1 1 100 10000 100
# qsub -t1-100:1 submit_ip_zd.py 0.8 2 0.1 1 100 10000 100
# qsub -t1-100:1 submit_ip_zd.py 0.9 2 0.1 1 100 10000 100
# qsub -t1-100:1 submit_ip_zd.py 1 2 0.1 1 100 10000 100
# qsub -t1-100:1 submit_ip_zd.py 0 0 0.05 1 100 10000 100
# qsub -t1-100:1 submit_ip_zd.py 0.5 0 0.05 1 100 10000 100
# qsub -t1-100:1 submit_ip_zd.py 0.8 0 0.05 1 100 10000 100
# qsub -t1-100:1 submit_ip_zd.py 0.9 0 0.05 1 100 10000 100
# qsub -t1-100:1 submit_ip_zd.py 1 0 0.05 1 100 10000 100

if __name__ == '__main__':
    make_popscore_figs()
    make_roc_fig()
    make_auc_fig()
