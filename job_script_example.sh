#!/bin/sh
### General options
#BSUB -q gpua100                     # specify GPU to run on
#BSUB -J hgt_gnn                     # job name
#BSUB -o hgt_gnn%J.out               # output name
#BSUB -e hgt_gnn%J.err		         # output name (error)
#BSUB -n 4                           # set number of cores
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"          # how much RAM to reserve
#BSUB -M 32GB                        # hard cap on RAM consumption
#BSUB -gpu "num=1:mode=exclusive_process" # number of GPU's & whether to share VRAM
#BSUB -W 24:00                      # Specify hours that the jub should run for

module purge
module load python3/3.12.9 # load python

source /work3/s204138/HGT/HGTenv/bin/activate # activate virtual env
cd /zhome/d6/f/156047/dtugraphsuite # go to directory with python file

## run scripts
python3 gnn_experiment.py --model=ogb_hgt_reproduction
python3 src/evaluate_gnn.py --config-name=ogb_hgt_reproduction_eval
