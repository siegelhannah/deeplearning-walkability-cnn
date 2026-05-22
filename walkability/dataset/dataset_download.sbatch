#!/bin/bash
#SBATCH --job-name=proj_dataset_download
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=compute
#SBATCH --account=dsci410_510



python dataset_building.py
