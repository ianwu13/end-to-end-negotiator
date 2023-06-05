# UTTERANCE LEVEL COMMANDS

## Supervised Learning
```
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python train.py  --data data/negotiate --cuda --bsz 16  --clip 0.5  --decay_every 1  --decay_rate 5.0  --dropout 0.5  --init_range 0.1  --lr 1  --max_epoch 30  --min_lr 0.01  --momentum 0.1  --nembed_ctx 64  --nembed_word 256  --nesterov  --nhid_attn 256  --nhid_ctx 64  --nhid_lang 128  --nhid_sel 256  --nhid_strat 128  --sel_weight 0.5  --model_file ../../logs/sv_model.pt
```

## Selfplay
```
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python selfplay.py --alice_model_file ../../logs/sv_model.pt --bob_model_file ../../logs/sv_model.pt --context_file data/negotiate/selfplay.txt --temperature 0.5 --log_file ../../logs/selfplay.log --ref_text data/negotiate/train.txt
```

## Human-Agent Chat
```
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python chat.py --model_file ../../logs/sv_model.pt --context_file data/negotiate/selfplay.txt --temperature 0.5 --ref_text data/negotiate/train.txt
```

## Reinforcement Learning
```
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python reinforce.py --cuda --data data/negotiate --bsz 16 --clip 1 --context_file data/negotiate/selfplay.txt --eps 0.0 --gamma 0.95 --lr 0.5 --momentum 0.1 --nepoch 4 --nesterov --ref_text data/negotiate/train.txt --rl_clip 1 --rl_lr 0.2 --score_threshold 6 --sv_train_freq 4 --temperature 0.5 --alice_model ../../logs/sv_model.pt --bob_model ../../logs/sv_model.pt --rw_type utility --output_model_file ../../logs/rl_model.pt
```



# DIALOGUE ACT LEVEL COMMANDS

## Supervised Learning
```
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python train.py  --data data/dia_act --cuda --bsz 16  --clip 0.5  --decay_every 1  --decay_rate 5.0  --dropout 0.5  --init_range 0.1  --lr 1  --max_epoch 30  --min_lr 0.01  --momentum 0.1  --nembed_ctx 64  --nembed_word 256  --nesterov  --nhid_attn 256  --nhid_ctx 64  --nhid_lang 128  --nhid_sel 256  --nhid_strat 128  --sel_weight 0.5  --model_file ../../logs/sv_model_da.pt
```

## Selfplay
```
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python selfplay.py --alice_model_file ../../logs/sv_model_da.pt --bob_model_file ../../logs/sv_model_da.pt --context_file data/dia_act/selfplay.txt --temperature 0.5 --log_file ../../logs/selfplay_da.log --ref_text data/dia_act/train.txt
```

## Human-Agent Chat
```
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python chat.py --model_file ../../logs/sv_model_da.pt --context_file data/dia_act/selfplay.txt --temperature 0.5 --ref_text data/dia_act/train.txt
```

## Reinforcement Learning
```
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python reinforce.py --cuda --data data/dia_act --bsz 16 --clip 1 --context_file data/dia_act/selfplay.txt --eps 0.0 --gamma 0.95 --lr 0.5 --momentum 0.1 --nepoch 4 --nesterov --ref_text data/dia_act/train.txt --rl_clip 1 --rl_lr 0.2 --score_threshold 6 --sv_train_freq 4 --temperature 0.5 --alice_model ../../logs/sv_model_da.pt --bob_model ../../logs/sv_model_da.pt --rw_type utility --output_model_file ../../logs/rl_model.pt
```








# OLD

corrected rl - v1

basically uses the corrected score_choices pipeline. where whenever the outputs of the two models mismatch - that is considered as a failure case, rather than a disagreement. this is just noisy.

The actual disagreement is the case where the models fail to come to an agreement - no_agreement -> in which case the models receive a reward of 0. - you can also make it -100 or something to give a high negative to such a behavior.


there is interesting promise - v1 ends the conversation. v1 vs v2 direct comparison doesn't make sense..only do it when comparing to a 3rd group - so compared to 3rd group and v1 is better across multiple metrics like joint value, etc.

v2

Decrease the reward further for no agreement case.

Is the outcome prediction model separate already in the next paper? how is agreement disagreement defined in that work? Separate out the outcome prediction model - keep track of failure cases...keep track of the outcome prediction points and confidence and entropy.


CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python bot_bot_play.py --models_dir ../../logs/ --conv_dir ../../logs/bot_bot/convs/ --context_file data/negotiate/test_dummy.txt --temperature 0.5 --ref_text data/negotiate/train.txt

conda activate spr23

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python reinforce_generic.py --cuda --data data/negotiate --bsz 16 --clip 1 --context_file data/negotiate/selfplay.txt --eps 0.0 --gamma 0.95 --lr 0.5 --momentum 0.1 --nesterov --ref_text data/negotiate/train.txt --rl_clip 1 --rl_lr 0.2 --score_threshold 6 --sv_train_freq 4 --temperature 0.5 --rw_type utility --output_model_file ../../logs/rl_model.pt --policy_model ../../logs/sv_model.pt --opp_models ../../logs/sv_model.pt,../../logs/rl_model_rw_utility_1_0_0_0.pt,../../logs/rl_model_rw_utility_1_0_-0.75_-0.75.pt --nepoch_per_opp 1 --num_opp_used 3

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python reinforce_generic.py --cuda --data data/negotiate --bsz 16 --clip 1 --context_file data/negotiate/selfplay.txt --eps 0.0 --gamma 0.95 --lr 0.5 --momentum 0.1 --nesterov --ref_text data/negotiate/train.txt --rl_clip 1 --rl_lr 0.2 --score_threshold 6 --sv_train_freq 4 --temperature 0.5 --rw_type utility --output_model_file ../../logs/rl_model.pt --policy_model ../../logs/sv_model.pt --opp_models ../../logs/sv_model.pt,../../logs/rl_model_rw_utility_1_0_0_0.pt,../../logs/rl_model_rw_utility_1_0_-0.75_-0.75.pt --nepoch_per_opp 1 --num_opp_used 9


CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python reinforce.py --cuda --data data/negotiate --bsz 16 --clip 1 --context_file data/negotiate/selfplay.txt --eps 0.0 --gamma 0.95 --lr 0.5 --momentum 0.1 --nepoch 4 --nesterov --ref_text data/negotiate/train.txt --rl_clip 1 --rl_lr 0.2 --score_threshold 6 --sv_train_freq 4 --temperature 0.5 --alice_model ../../logs/sv_model.pt --bob_model ../../logs/rl_model_rw_utility_1_0_0_0.pt --rw_type utility --output_model_file ../../logs/rl_model.pt --rw_type own_points


CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python bot_bot_play.py --models_dir ../../logs/ --conv_dir ../../logs/bot_bot/convs/ --context_file data/negotiate/test.txt --temperature 0.5 --ref_text data/negotiate/train.txt