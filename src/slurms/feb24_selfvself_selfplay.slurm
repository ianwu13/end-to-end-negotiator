#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --account=glucas_540

#SBATCH --output=R-%x.%j.out

cd ..

# setup
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate e2e


echo "SELFSELF V SELFSELF"
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python3 selfplay.py --alice_model_file ../logs/jan24_casino_custform_rl_selfish_v_selfish_4ep_rw_own_points.pt --bob_model_file ../logs/jan24_casino_custform_rl_selfish_v_selfish_4ep_rw_own_points.pt --context_file data/fd_r2_jan_2024/selfplay.txt --temperature 0.5 --log_file ../../logs/sp_jan2024_sup_v_selfvself.log --ref_text data/fd_r2_jan_2024/train.txt


echo "SELFSELF V SELFSELF Fixed"
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python3 selfplay.py --alice_model_file ../logs/jan24_casino_custform_rl_selfish_v_selfish_4ep_fixed_rw_own_points.pt --bob_model_file ../logs/jan24_casino_custform_rl_selfish_v_selfish_4ep_fixed_rw_own_points.pt --context_file data/fd_r2_jan_2024_fixed/selfplay.txt --temperature 0.5 --log_file ../../logs/sp_jan2024_fixed_data_sup_v_selfvself.log --ref_text data/fd_r2_jan_2024_fixed/train.txt
