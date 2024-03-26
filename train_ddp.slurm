#!/bin/bash -l

#SBATCH --job-name=train_style_classification_ddp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2
#SBATCH --time=2:00:00
#SBATCH --output=./outslurm_ddp/%j-slurm.out
#SBATCH --constraint=v100
#SBATCH --mem=200G

module load machine_learning
srun ./script/train.sh
