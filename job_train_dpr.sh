#!/bin/sh
### General options
#BSUB -q gpuv100                     # cua de GPU (mira 'bqueues'; si està plena prova gpua100)
#BSUB -J dpr_train                   # nom del job
#BSUB -o dpr_train_%J.out            # stdout: prints del trainer (tail -f per seguir-ho)
#BSUB -e dpr_train_%J.err            # stderr: errors i tracebacks
#BSUB -n 4                           # nuclis de CPU
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"          # RAM a reservar
#BSUB -M 16GB                        # límit dur de RAM
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00                        # temps màxim (hh:mm); el model és petit, 4h sobra

module purge
module load python3/3.12.9

source $HOME/dpr_env/bin/activate            # venv creat al pas 5
cd $HOME/02456-retrieval-project              # arrel del projecte (on hi ha src/ i config/)

nvidia-smi                           # queda al .out: confirma quina GPU t'han donat

# -u: sense buffer, perquè els prints apareguin al .out mentre entrena
python3 -u train_dpr.py
