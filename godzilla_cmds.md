# Validation Set Testing
```
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python train.py  --data data/val_dnd --cuda --bsz 16  --clip 0.5  --decay_every 1  --decay_rate 5.0  --dropout 0.5  --init_range 0.1  --lr 1  --max_epoch 30  --min_lr 0.01  --momentum 0.1  --nembed_ctx 64  --nembed_word 256  --nesterov  --nhid_attn 256  --nhid_ctx 64  --nhid_lang 128  --nhid_sel 256  --nhid_strat 128  --sel_weight 0.5  --model_file ../logs/VAL_dnd_supervised_30ep.pt

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python reinforce.py --cuda --data data/val_dnd --bsz 16 --clip 1 --context_file data/val_dnd/selfplay_select.txt --eps 0.0 --gamma 0.95 --lr 0.5 --momentum 0.1 --nepoch 4 --nesterov --ref_text data/val_dnd/train.txt --rl_clip 1 --rl_lr 0.2 --score_threshold 6 --sv_train_freq 4 --temperature 0.5 --alice_model ../logs/VAL_dnd_supervised_30ep.pt --bob_model ../logs/VAL_dnd_supervised_30ep.pt --rw_type own_points --output_model_file ../logs/VAL_dnd_rl_selfish_4ep.pt
```

```
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python train.py  --data data/val_casino_dndformat --cuda --bsz 16  --clip 0.5  --decay_every 1  --decay_rate 5.0  --dropout 0.5  --init_range 0.1  --lr 1  --max_epoch 30  --min_lr 0.01  --momentum 0.1  --nembed_ctx 64  --nembed_word 256  --nesterov  --nhid_attn 256  --nhid_ctx 64  --nhid_lang 128  --nhid_sel 256  --nhid_strat 128  --sel_weight 0.5  --model_file ../logs/VAL_casino_dndformat_supervised_30ep.pt

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python reinforce.py --cuda --data data/val_casino_dndformat --bsz 16 --clip 1 --context_file data/val_casino_dndformat/selfplay_select.txt --eps 0.0 --gamma 0.95 --lr 0.5 --momentum 0.1 --nepoch 4 --nesterov --ref_text data/val_casino_dndformat/train.txt --rl_clip 1 --rl_lr 0.2 --score_threshold 6 --sv_train_freq 4 --temperature 0.5 --alice_model ../logs/VAL_casino_dndformat_supervised_30ep.pt --bob_model ../logs/VAL_casino_dndformat_supervised_30ep.pt --rw_type own_points --output_model_file ../logs/VAL_casino_dndformat_rl_selfish_4ep.pt
```

```
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python train.py  --data data/val_casino_custformat --cuda --bsz 16  --clip 0.5  --decay_every 1  --decay_rate 5.0  --dropout 0.5  --init_range 0.1  --lr 1  --max_epoch 30  --min_lr 0.01  --momentum 0.1  --nembed_ctx 64  --nembed_word 256  --nesterov  --nhid_attn 256  --nhid_ctx 64  --nhid_lang 128  --nhid_sel 256  --nhid_strat 128  --sel_weight 0.5  --model_file ../logs/VAL_casino_custformat_supervised_30ep.pt
```

# RL training commands

## dnd
```
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python train.py  --data data/final_dnd --cuda --bsz 16  --clip 0.5  --decay_every 1  --decay_rate 5.0  --dropout 0.5  --init_range 0.1  --lr 1  --max_epoch 30  --min_lr 0.01  --momentum 0.1  --nembed_ctx 64  --nembed_word 256  --nesterov  --nhid_attn 256  --nhid_ctx 64  --nhid_lang 128  --nhid_sel 256  --nhid_strat 128  --sel_weight 0.5  --model_file ../logs/dnd_supervised_30ep.pt

echo SELFISH

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python reinforce.py --cuda --data data/final_dnd --bsz 16 --clip 1 --context_file data/final_dnd/selfplay.txt --eps 0.0 --gamma 0.95 --lr 0.5 --momentum 0.1 --nepoch 4 --nesterov --ref_text data/final_dnd/train.txt --rl_clip 1 --rl_lr 0.2 --score_threshold 6 --sv_train_freq 4 --temperature 0.5 --alice_model ../logs/dnd_supervised_30ep.pt --bob_model ../logs/dnd_supervised_30ep.pt --rw_type own_points --output_model_file ../logs/dnd_rl_selfish_4ep.pt

echo FAIR

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python reinforce.py --cuda --data data/final_dnd --bsz 16 --clip 1 --context_file data/final_dnd/selfplay.txt --eps 0.0 --gamma 0.95 --lr 0.5 --momentum 0.1 --nepoch 4 --nesterov --ref_text data/final_dnd/train.txt --rl_clip 1 --rl_lr 0.2 --score_threshold 6 --sv_train_freq 4 --temperature 0.5 --alice_model ../logs/dnd_supervised_30ep.pt --bob_model ../logs/dnd_supervised_30ep.pt --rw_type combine50_50 --output_model_file ../logs/dnd_rl_fair_4ep.pt
```

## casino dnd format
```
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python train.py  --data data/final_casino_dndformat --cuda --bsz 16  --clip 0.5  --decay_every 1  --decay_rate 5.0  --dropout 0.5  --init_range 0.1  --lr 1  --max_epoch 30  --min_lr 0.01  --momentum 0.1  --nembed_ctx 64  --nembed_word 256  --nesterov  --nhid_attn 256  --nhid_ctx 64  --nhid_lang 128  --nhid_sel 256  --nhid_strat 128  --sel_weight 0.5  --model_file ../logs/casino_dndformat_supervised_30ep.pt

echo SELFISH

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python reinforce.py --cuda --data data/final_casino_dndformat --bsz 16 --clip 1 --context_file data/final_casino_dndformat/selfplay.txt --eps 0.0 --gamma 0.95 --lr 0.5 --momentum 0.1 --nepoch 4 --nesterov --ref_text data/final_casino_dndformat/train.txt --rl_clip 1 --rl_lr 0.2 --score_threshold 6 --sv_train_freq 4 --temperature 0.5 --alice_model ../logs/casino_dndformat_supervised_30ep.pt --bob_model ../logs/casino_dndformat_supervised_30ep.pt --rw_type own_points --output_model_file ../logs/casino_dndform_rl_selfish_4ep.pt

echo FAIR

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python reinforce.py --cuda --data data/final_casino_dndformat --bsz 16 --clip 1 --context_file data/final_casino_dndformat/selfplay.txt --eps 0.0 --gamma 0.95 --lr 0.5 --momentum 0.1 --nepoch 4 --nesterov --ref_text data/final_casino_dndformat/train.txt --rl_clip 1 --rl_lr 0.2 --score_threshold 6 --sv_train_freq 4 --temperature 0.5 --alice_model ../logs/casino_dndformat_supervised_30ep.pt --bob_model ../logs/casino_dndformat_supervised_30ep.pt --rw_type combine50_50 --output_model_file ../logs/casino_dndform_rl_fair_4ep.pt
```

## casino custom format
```
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python train.py  --data data/final_casino_custformat --cuda --bsz 16  --clip 0.5  --decay_every 1  --decay_rate 5.0  --dropout 0.5  --init_range 0.1  --lr 1  --max_epoch 30  --min_lr 0.01  --momentum 0.1  --nembed_ctx 64  --nembed_word 256  --nesterov  --nhid_attn 256  --nhid_ctx 64  --nhid_lang 128  --nhid_sel 256  --nhid_strat 128  --sel_weight 0.5  --model_file ../logs/casino_custformat_supervised_30ep.pt

echo SELFISH

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python reinforce.py --cuda --data data/final_casino_custformat --bsz 16 --clip 1 --context_file data/final_casino_custformat/selfplay.txt --eps 0.0 --gamma 0.95 --lr 0.5 --momentum 0.1 --nepoch 4 --nesterov --ref_text data/final_casino_custformat/train.txt --rl_clip 1 --rl_lr 0.2 --score_threshold 6 --sv_train_freq 4 --temperature 0.5 --alice_model ../logs/casino_custformat_supervised_30ep.pt --bob_model ../logs/casino_custformat_supervised_30ep.pt --rw_type own_points --output_model_file ../logs/casino_custform_rl_selfish_4ep.pt

echo FAIR

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python reinforce.py --cuda --data data/final_casino_custformat --bsz 16 --clip 1 --context_file data/final_casino_custformat/selfplay.txt --eps 0.0 --gamma 0.95 --lr 0.5 --momentum 0.1 --nepoch 4 --nesterov --ref_text data/final_casino_custformat/train.txt --rl_clip 1 --rl_lr 0.2 --score_threshold 6 --sv_train_freq 4 --temperature 0.5 --alice_model ../logs/casino_custformat_supervised_30ep.pt --bob_model ../logs/casino_custformat_supervised_30ep.pt --rw_type combine50_50 --output_model_file ../logs/casino_custform_rl_fair_4ep.pt
```



# UTTERANCE LEVEL COMMANDS

## Supervised Learning
```
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python train.py  --data data/negotiate --cuda --bsz 16  --clip 0.5  --decay_every 1  --decay_rate 5.0  --dropout 0.5  --init_range 0.1  --lr 1  --max_epoch 30  --min_lr 0.01  --momentum 0.1  --nembed_ctx 64  --nembed_word 256  --nesterov  --nhid_attn 256  --nhid_ctx 64  --nhid_lang 128  --nhid_sel 256  --nhid_strat 128  --sel_weight 0.5  --model_file ../logs/sv_model_30ep_dndNegotiate.pt
```

## Selfplay
```
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python selfplay.py --alice_model_file ../../logs/sv_model.pt --bob_model_file ../../logs/sv_model.pt --context_file data/negotiate/selfplay.txt --temperature 0.5 --log_file ../../logs/selfplay.log --ref_text data/negotiate/train.txt
```

## Human-Agent Chat
```
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python chat.py --model_file ../../logs/sv_model.pt --context_file data/negotiate/selfplay.txt --temperature 0.5 --ref_text data/negotiate/train.txt --log_file ../../logs/chat_test.log
```

## Reinforcement Learning
```
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python reinforce.py --cuda --data data/negotiate --bsz 16 --clip 1 --context_file data/negotiate/selfplay.txt --eps 0.0 --gamma 0.95 --lr 0.5 --momentum 0.1 --nepoch 1 --nesterov --ref_text data/negotiate/train.txt --rl_clip 1 --rl_lr 0.2 --score_threshold 6 --sv_train_freq 4 --temperature 0.5 --alice_model ../../logs/sv_model.pt --bob_model ../../logs/sv_model.pt --rw_type utility --output_model_file ../../logs/rl_model.pt
```



# DIALOGUE ACT LEVEL COMMANDS

## Supervised Learning
```
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python train.py  --data data/dia_act_doub_prop --cuda --bsz 16  --clip 0.5  --decay_every 1  --decay_rate 5.0  --dropout 0.5  --init_range 0.1  --lr 1  --max_epoch 30  --min_lr 0.01  --momentum 0.1  --nembed_ctx 64  --nembed_word 256  --nesterov  --nhid_attn 256  --nhid_ctx 64  --nhid_lang 128  --nhid_sel 256  --nhid_strat 128  --sel_weight 0.5  --model_file ../../logs/sv_model_dia_act_doub_prop.pt
```

## Selfplay
```
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python selfplay.py --alice_model_file ../../logs/sv_model_dia_act_doub_prop.pt --bob_model_file ../../logs/sv_model_dia_act_doub_prop.pt --context_file data/dia_act_doub_prop/selfplay.txt --temperature 0.5 --log_file ../../logs/selfplay_dia_act_doub_prop.log --ref_text data/dia_act_doub_prop/train.txt
```

## Human-Agent Chat
```
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python chat.py --model_file ../../logs/sv_model_dia_act_doub_prop.pt --context_file data/dia_act_doub_prop/selfplay.txt --temperature 0.5 --ref_text data/dia_act_doub_prop/train.txt
```

## Reinforcement Learning
```
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python reinforce.py --cuda --data data/dia_act_doub_prop --bsz 16 --clip 1 --context_file data/dia_act_doub_prop/selfplay.txt --eps 0.0 --gamma 0.95 --lr 0.5 --momentum 0.1 --nepoch 4 --nesterov --ref_text data/dia_act_doub_prop/train.txt --rl_clip 1 --rl_lr 0.2 --score_threshold 6 --sv_train_freq 4 --temperature 0.5 --alice_model ../../logs/sv_model_dia_act_doub_prop.pt --bob_model ../../logs/sv_model_dia_act_doub_prop.pt --rw_type utility --output_model_file ../../logs/rl_model_dia_act_doub_prop.pt
```



# DND DIALOGUE ACT RL EXPERIMENT COMMANDS

## Maximize Self Points
```
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python reinforce.py --cuda --data data/dia_act_doub_prop --bsz 16 --clip 1 --context_file data/dia_act_doub_prop/selfplay.txt --eps 0.0 --gamma 0.95 --lr 0.5 --momentum 0.1 --nepoch 4 --nesterov --ref_text data/dia_act_doub_prop/train.txt --rl_clip 1 --rl_lr 0.2 --score_threshold 6 --sv_train_freq 4 --temperature 0.5 --alice_model ../../logs/sv_model_dia_act_doub_prop.pt --bob_model ../../logs/sv_model_dia_act_doub_prop.pt --rw_type own_points --output_model_file ../../logs/rl_model_self_pts.pt

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python selfplay.py --alice_model_file ../../logs/rl_model_self_pts_rw_own_points.pt --bob_model_file ../../logs/sv_model_dia_act_doub_prop.pt --context_file data/dia_act_doub_prop/selfplay.txt --temperature 0.5 --ref_text data/dia_act_doub_prop/train.txt --log_file ../../logs/selfplay_rl_self_pts.log
```

## Maximize Partner Points
```
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python reinforce.py --cuda --data data/dia_act_doub_prop --bsz 16 --clip 1 --context_file data/dia_act_doub_prop/selfplay.txt --eps 0.0 --gamma 0.95 --lr 0.5 --momentum 0.1 --nepoch 4 --nesterov --ref_text data/dia_act_doub_prop/train.txt --rl_clip 1 --rl_lr 0.2 --score_threshold 6 --sv_train_freq 4 --temperature 0.5 --alice_model ../../logs/sv_model_dia_act_doub_prop.pt --bob_model ../../logs/sv_model_dia_act_doub_prop.pt --rw_type partner_points --output_model_file ../../logs/rl_model_prtnr_pts.pt

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python selfplay.py --alice_model_file ../../logs/rl_model_prtnr_pts_rw_partner_points.pt --bob_model_file ../../logs/sv_model_dia_act_doub_prop.pt --context_file data/dia_act_doub_prop/selfplay.txt --temperature 0.5 --ref_text data/dia_act_doub_prop/train.txt --log_file ../../logs/selfplay_rl_prtnr_pts.log
```

## Maximize Combine50_50
```
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python reinforce.py --cuda --data data/dia_act_doub_prop --bsz 16 --clip 1 --context_file data/dia_act_doub_prop/selfplay.txt --eps 0.0 --gamma 0.95 --lr 0.5 --momentum 0.1 --nepoch 4 --nesterov --ref_text data/dia_act_doub_prop/train.txt --rl_clip 1 --rl_lr 0.2 --score_threshold 6 --sv_train_freq 4 --temperature 0.5 --alice_model ../../logs/sv_model_dia_act_doub_prop.pt --bob_model ../../logs/sv_model_dia_act_doub_prop.pt --rw_type combine50_50 --output_model_file ../../logs/rl_model_5050_pts.pt

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python selfplay.py --alice_model_file ../../logs/rl_model_5050_pts_rw_combine50_50.pt --bob_model_file ../../logs/sv_model_dia_act_doub_prop.pt --context_file data/dia_act_doub_prop/selfplay.txt --temperature 0.5 --ref_text data/dia_act_doub_prop/train.txt --log_file ../../logs/selfplay_rl_5050_pts.log
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