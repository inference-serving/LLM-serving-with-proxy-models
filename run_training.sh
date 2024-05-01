#!/bin/bash
#SBATCH --job-name=training
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=2-00:00:00
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --partition=k2-gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s.ghafouri@qub.ac.uk

# Load CUDA module
module purge
module load libs/nvidia-cuda/12.3.0/bin

# Load Python module
module load apps/python3/3.10.5/gcc-9.3.0

# Create and activate central virtual environment
python3 -m venv central
source central/bin/activate

# Install required packages
pip3 install -r requirements.txt

# Run the preprocessing task
cd ./output-token-len-predictions
python preprocess_dataset.py

# Run the training task
python latency_prediction.py

# Deactivate virtual environment
deactivate