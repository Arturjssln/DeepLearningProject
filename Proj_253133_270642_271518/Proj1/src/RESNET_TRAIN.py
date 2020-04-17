import subprocess
import os


PYTHON_INTER = 'python' #change accordingly to your environment 

NB_RESIDUAL_BLOCKS_RANGE = [1, 2, 3, 4, 5]
NB_CHANNELS_RANGE = [1, 2, 4, 8, 16]
KERNEL_SIZE_RANGE = [3, 5, 7]

RESIDUAL = True
BN = True

FNULL = open(os.devnull, 'w')

nb_eval = len(NB_RESIDUAL_BLOCKS_RANGE) * len(NB_CHANNELS_RANGE) * len(KERNEL_SIZE_RANGE)
eval_ = 0
print('* Number of evaluations : {}'.format(nb_eval))
for i in NB_RESIDUAL_BLOCKS_RANGE:
    for j in NB_CHANNELS_RANGE:
        for k in KERNEL_SIZE_RANGE:
            eval_ += 1
            params = '--architecture resnet --save_fig --nb_residual_blocks {0} --nb_channels {1} --kernel_size {2} --force_axis --datasize 350'.format(i, j, k)
            if BN :
                params += ' --bn'
            if RESIDUAL :
                params += ' --residual'
            print(params)
            subprocess.call('{0} dev.py {1}'.format(PYTHON_INTER, params), shell=True, stdout=FNULL)
            print('** Evaluation #{} Done ({:.1f}% overall)'.format(eval_, eval_*100/nb_eval))