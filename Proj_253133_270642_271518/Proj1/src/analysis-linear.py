import subprocess
import os
import sys


NB_NODES = [16, 32, 64, 128, 216]
DROPOUT = [True, False]
DEEP = [True, False]


FNULL = open(os.devnull, 'w')

nb_eval = len(NB_NODES) * len(DROPOUT) * len(DEEP)
eval_ = 0
print('* Number of evaluations : {}'.format(nb_eval))
for i in NB_NODES:
    for dropout in DROPOUT:
        for deep in DEEP:
            eval_ += 1
            params = '--architecture linear --save_fig --nodes {0} --force_axis'.format(i)
            if deep :
                params += ' --deep'
            if dropout :
                params += ' --dropout'
            print(params)
            subprocess.call('{0} dev.py {1}'.format(sys.executable, params), shell=True, stdout=FNULL)
            print('** Evaluation #{} Done ({:.1f}% overall)'.format(eval_, eval_*100/nb_eval))
