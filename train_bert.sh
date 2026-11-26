#!/bin/bash
#BSUB -J dpr_bert
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 08:00
#BSUB -o logs/train_bert_%J.out
#BSUB -e logs/train_bert_%J.err

module load python3/3.11.9
module load cuda/12.1
source /dtu/blackhole/1e/233438/venvs/dpr/bin/activate

cd /dtu/blackhole/1e/233438/02456-retrieval-project
python -u train_bert.py
