#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --account=glucas_540

#SBATCH --output=R-%x.%j.out

cd ..

# setup
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate e2e

echo "SELFPLAY JAN 2024 DATA"

echo "SUPERVISED V SUPERVISED"
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python3 selfplay.py --alice_model_file models/model_files/fd_jan24/jan24_casino_custformat_supervised_30ep.pt --bob_model_file models/model_files/fd_jan24/jan24_casino_custformat_supervised_30ep.pt --context_file data/fd_r2_jan_2024/selfplay.txt --temperature 0.5 --log_file ../../logs/sp_jan2024_sup_v_sup.log --ref_text data/fd_r2_jan_2024/train.txt

echo "SELFISH V SELFISH"
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python3 selfplay.py --alice_model_file models/model_files/fd_jan24/jan24_casino_custform_rl_selfish_4ep_rw_own_points.pt --bob_model_file models/model_files/fd_jan24/jan24_casino_custform_rl_selfish_4ep_rw_own_points.pt --context_file data/fd_r2_jan_2024/selfplay.txt --temperature 0.5 --log_file ../../logs/sp_jan2024_selfish_v_selfish.log --ref_text data/fd_r2_jan_2024/train.txt

echo "FAIR V FAIR"
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python3 selfplay.py --alice_model_file models/model_files/fd_jan24/jan24_casino_custform_rl_fair_4ep_rw_combine50_50.pt --bob_model_file models/model_files/fd_jan24/jan24_casino_custform_rl_fair_4ep_rw_combine50_50.pt --context_file data/fd_r2_jan_2024/selfplay.txt --temperature 0.5 --log_file ../../logs/sp_jan2024_fair_v_fair.log --ref_text data/fd_r2_jan_2024/train.txt

echo "SUPERVISED V SELFISH"
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python3 selfplay.py --alice_model_file models/model_files/fd_jan24/jan24_casino_custformat_supervised_30ep.pt --bob_model_file models/model_files/fd_jan24/jan24_casino_custform_rl_selfish_4ep_rw_own_points.pt --context_file data/fd_r2_jan_2024/selfplay.txt --temperature 0.5 --log_file ../../logs/sp_jan2024_sup_v_selfish.log --ref_text data/fd_r2_jan_2024/train.txt

echo "SUPERVISED V FAIR"
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python3 selfplay.py --alice_model_file models/model_files/fd_jan24/jan24_casino_custformat_supervised_30ep.pt --bob_model_file models/model_files/fd_jan24/jan24_casino_custform_rl_fair_4ep_rw_combine50_50.pt --context_file data/fd_r2_jan_2024/selfplay.txt --temperature 0.5 --log_file ../../logs/sp_jan2024_sup_v_fair.log --ref_text data/fd_r2_jan_2024/train.txt

echo "SELFISH V FAIR"
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python3 selfplay.py --alice_model_file models/model_files/fd_jan24/jan24_casino_custform_rl_selfish_4ep_rw_own_points.pt --bob_model_file models/model_files/fd_jan24/jan24_casino_custform_rl_fair_4ep_rw_combine50_50.pt --context_file data/fd_r2_jan_2024/selfplay.txt --temperature 0.5 --log_file ../../logs/sp_jan2024_selfish_v_fair.log --ref_text data/fd_r2_jan_2024/train.txt