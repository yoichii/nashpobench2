#!/bin/bash
#$ -l rt_G.small=1
#$ -l h_rt=3:00:00
#$ -o stdouts
#$ -e stderrs
#$ -t 1-14400
#$ -cwd
#$ -m e

# set up env
source ~/.bash_profile
source /etc/profile.d/modules.sh
module load gcc/7.4.0 cuda/10.1/10.1.243 cudnn/7.6/7.6.5
conda activate py376
source venv/bin/activate

# variables
seeds=(222 555 888)
N_SEED=3
N_HP=48
N_SETTING=$(( $N_SEED * $N_HP ))
cellidx=$(( ($SGE_TASK_ID-1) / $N_SETTING ))
codeidx=$(( ( ($SGE_TASK_ID-1) / $N_SEED ) % $N_HP ))
seed=$(( seeds[ ($SGE_TASK_ID-1) % $N_SEED ] ))

# run
python train.py\
    training.epochs=200\
    architecture.cellidx=$cellidx\
    training.codesidx=$codeidx\
    training.seed=$seed\
    --config-name "acml200"
