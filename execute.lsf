#!/bin/bash

#BSUB -J JOBNAME
#BSUB -n 4
#BSUB -o %J.stdout
#BSUB -e %J.stderr
#BSUB -R "span[ptile=4]"
#BSUB -W 2400:00
#BSUB -q q2080
#BSUB -gpu num=4

## Loading modules
module load gcc/8.3.0
module load zlib/1.2.11-gcc-8.3.0
module load mpi/intel/2019.5
module load python/anaconda3/2019.10
module load cuda/10.1
module load git/2.18.0

## Setup User Environment
HOME=/share/home/grp-lizy/pangyx
CONDA_VIRTUAL_ENV=AMPBertMT
CONDA_ROOT=/share/apps/anaconda3/2019.10

source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $CONDA_VIRTUAL_ENV

## First mission training
python train.py  --cuda --seed 810 --task "AMP" --lr 0.03 --ckpt-iter 60 -e 256 -b 64 -d "test_first-stage-(neg,pos)-(2,2)"
## Second mission training
python train.py  --cuda --seed 514 --task "mtl" --lr 0.04 --ckpt-iter 60 -e 300 -b 64 -d "mtl-(neg,pos)-(10,2)"
## Evaluation
python evaluate.py --path "/share/home/grp-lizy/pangyx/Experiments/AMPBertMT/results/first-stage-(neg,pos)-(3,1)" --cuda True