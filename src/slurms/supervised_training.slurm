#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --account=glucas_540

cd ..

# setup
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate e2e

# Run script
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python train.py  --data data/negotiate --cuda --bsz 16  --clip 0.5  --decay_every 1  --decay_rate 5.0  --dropout 0.5  --init_range 0.1  --lr 1  --max_epoch 1  --min_lr 0.01  --momentum 0.1  --nembed_ctx 64  --nembed_word 256  --nesterov  --nhid_attn 256  --nhid_ctx 64  --nhid_lang 128  --nhid_sel 256  --nhid_strat 128  --sel_weight 0.5  --model_file ../logs/sv_model.pt
