#!/bin/bash
#SBATCH --job-name=walkability_train_compare
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=gpu
#SBATCH --account=dsci410_510
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

source ~/.bashrc
conda activate walkability
cd /home/hsiegel/deeplearning_finalproj/deeplearning-walkability-cnn/walkability

mkdir -p logs

python train_compare_models.py
