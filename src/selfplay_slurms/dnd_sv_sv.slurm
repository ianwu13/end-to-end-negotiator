#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --account=glucas_540
#SBATCH --output=dnd_sv_sv.out

cd ..

# setup
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate e2e

# Run script
python3 selfplay.py --alice_model_file models/model_files/dnd_supervised_30ep.pt --bob_model_file models/model_files/dnd_supervised_30ep.pt --context_file data/final_dnd_valid/selfplay.txt --ref_text data/final_casino_dndformat/train.txt --log_file ../logs/dnd_sv_sv.txt
