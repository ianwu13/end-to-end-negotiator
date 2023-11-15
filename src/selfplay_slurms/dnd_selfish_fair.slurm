#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --account=glucas_540
#SBATCH --output=dnd_selfish_fair.out

cd ..

# setup
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate e2e

# Run script
python3 selfplay.py --alice_model_file models/model_files/dnd_rl_selfish_4ep_rw_own_points.pt --bob_model_file models/model_files/dnd_rl_fair_4ep_rw_combine50_50.pt --context_file data/final_dnd_valid/selfplay.txt --ref_text data/final_casino_dndformat/train.txt --log_file ../logs/dnd_selfish_fair.txt
