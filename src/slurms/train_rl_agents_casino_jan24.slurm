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

# Run script
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python train.py  --data data/fd_r2_jan_2024 --cuda --bsz 16  --clip 0.5  --decay_every 1  --decay_rate 5.0  --dropout 0.5  --init_range 0.1  --lr 1  --max_epoch 30  --min_lr 0.01  --momentum 0.1  --nembed_ctx 64  --nembed_word 256  --nesterov  --nhid_attn 256  --nhid_ctx 64  --nhid_lang 128  --nhid_sel 256  --nhid_strat 128  --sel_weight 0.5  --model_file ../logs/jan24_casino_custformat_supervised_30ep.pt

echo SELFISH

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python reinforce.py --cuda --data data/fd_r2_jan_2024 --bsz 16 --clip 1 --context_file data/fd_r2_jan_2024/selfplay.txt --eps 0.0 --gamma 0.95 --lr 0.5 --momentum 0.1 --nepoch 4 --nesterov --ref_text data/fd_r2_jan_2024/train.txt --rl_clip 1 --rl_lr 0.2 --score_threshold 6 --sv_train_freq 4 --temperature 0.5 --alice_model ../logs/jan24_casino_custformat_supervised_30ep.pt --bob_model ../logs/jan24_casino_custformat_supervised_30ep.pt --rw_type own_points --output_model_file ../logs/jan24_casino_custform_rl_selfish_4ep.pt

echo FAIR

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python reinforce.py --cuda --data data/fd_r2_jan_2024 --bsz 16 --clip 1 --context_file data/fd_r2_jan_2024/selfplay.txt --eps 0.0 --gamma 0.95 --lr 0.5 --momentum 0.1 --nepoch 4 --nesterov --ref_text data/fd_r2_jan_2024/train.txt --rl_clip 1 --rl_lr 0.2 --score_threshold 6 --sv_train_freq 4 --temperature 0.5 --alice_model ../logs/jan24_casino_custformat_supervised_30ep.pt --bob_model ../logs/jan24_casino_custformat_supervised_30ep.pt --rw_type combine50_50 --output_model_file ../logs/jan24_casino_custform_rl_fair_4ep.pt
