import subprocess
import os
import sys


KERNEL_SIZE_RANGE = [3]
BN = [True, False]
DROPOUT = [True, False]
AUX_LOSS = [True, False]

FNULL = open(os.devnull, 'w')

nb_eval = len(BN) * len(DROPOUT) * len(KERNEL_SIZE_RANGE) * len(AUX_LOSS)
eval_ = 0
print('* Number of evaluations : {}'.format(nb_eval))
for i in KERNEL_SIZE_RANGE:
    for j in BN:
        for k in DROPOUT:
            for l in DROPOUT:
                eval_ += 1
                params = '--architecture lenet --save_fig --kernel_size {} --force_axis --datasize 500'.format(i)
                if j: #BN
                    params += ' --bn'
                if k: #dropout
                    params += ' --dropout'
                if l: #aux loss
                    params += ' --auxloss'
                print(params)
                subprocess.call('{0} dev.py {1}'.format(sys.executable, params), shell=True, stdout=FNULL)
                print('** Evaluation #{} Done ({:.1f}% overall)'.format(eval_, eval_*100/nb_eval))
