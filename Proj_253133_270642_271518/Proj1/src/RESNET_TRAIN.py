import subprocess
import os


PYTHON_INTER = 'python-3-7' #change accordingly to your environment

#NB_RESIDUAL_BLOCKS_RANGE = [1, 2, 3, 4, 5]
#NB_CHANNELS_RANGE = [1, 2, 4, 8, 16]
KERNEL_SIZE_RANGE = [3, 5]

#RESIDUAL = True
BN = [True, False]
DROPOUT = [True, False]

FNULL = open(os.devnull, 'w')

nb_eval = len(BN) * len(DROPOUT) * len(KERNEL_SIZE_RANGE)
eval_ = 0
print('* Number of evaluations : {}'.format(nb_eval))
for i in KERNEL_SIZE_RANGE:
    for j in BN:
        for k in DROPOUT:
            eval_ += 1
            params = '--architecture lenet --save_fig --kernel_size {} --force_axis --datasize 350'.format(i)
            if j: #BN
                params += ' --bn'
            if k: #dropout
                params += ' --dropout'
            print(params)
            subprocess.call('{0} dev.py {1}'.format(PYTHON_INTER, params), shell=True, stdout=FNULL)
            print('** Evaluation #{} Done ({:.1f}% overall)'.format(eval_, eval_*100/nb_eval))
