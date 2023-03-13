# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Enable Bot-Bot play in an nC2 manner and save all the logs with conv-level metrics.

Metrics:

Diversity, self-points, partner-points, joint-points, failure, agreement_rate_wa, agreement_rate_len, agreement_rate_total, pareto_optimal

# what all do I want to look at?

variables:
subset -> show for all subsets together..["all", "distributive", "integrative"]
metric = in Diversity, self-points, partner-points, joint-points, failure, agreement_rate_wa, agreement_rate_len, agreement_rate_total, pareto_optimal.
mode = ["overall", "pairwise"]

overall mode: will show an avg. performance over all the models in a table.
pairwise mode: will plot heatmaps.

Overall performance of every model for every metric (individual + joint) - avg over all the pairs.

Given a metric, a 2D Heatmap of pairwise performance.

-----------

Questions that you can ask:

1) Given two models, who is better when they play against each other? -> individual metrics at a pair level.

2) Overall, what model is better? -> avg of 1) scores over all model pairs where that model is involved - this is like the avg performance against different kinds of opponents.

3) Given a joint metric, which model pair is better? -> heatmaps for a joint metric?

Joint metrics

- Conv. length
- joint points
- failure %
- agreement_rate_wa
- agreement_rate_len
- agreement_rate_total
- pareto_optimal

Individual metrics
- All joint metrics
- uniqueness in sentences
- sent length.
- Self-points
- Partner-points

Good, then what to compute from each conversation:
- points for each of them
- entire conv.
- contexts - to decide distributive vs integrative.
- failure or not?
- no agreement due to length?
- no agreement due to walk away?
- is there agreement?
- pareto optimal agreement?
"""

import argparse
import json
import os
from tqdm import tqdm

from agent import LstmAgent, LstmRolloutAgent, BatchedRolloutAgent
import utils
from utils import DNDContextGenerator
from dialog import Dialog, DialogLogger
from models.dialog_model import DialogModel

def is_pareto_optimal(obj):
    """Check whether the agreed deal is optimal."""

    def compute_score(vals, picks):
        """Compute the score of the selection."""
        assert len(vals) == len(picks)
        return sum([v * p for v, p in zip(vals, picks)])

    def gen_choices(cnts, idx=0, choice=[]):
        """Generate all the valid choices.
        It generates both yours and your opponent choices.
        """
        if idx >= len(cnts):
            return [(choice[:], [n - c for n, c in zip(cnts, choice)]),]
        choices = []
        for c in range(cnts[idx] + 1):
            choice.append(c)
            choices += gen_choices(cnts, idx + 1, choice)
            choice.pop()
        return choices
    
    names = sorted(list(obj["ctxs"].keys()))
    cnts = [
        int(obj["ctxs"][names[0]][0]),
        int(obj["ctxs"][names[0]][2]),
        int(obj["ctxs"][names[0]][4]),
    ]

    vals1 = [
        int(obj["ctxs"][names[0]][1]),
        int(obj["ctxs"][names[0]][3]),
        int(obj["ctxs"][names[0]][5]),
    ]

    vals2 = [
        int(obj["ctxs"][names[1]][1]),
        int(obj["ctxs"][names[1]][3]),
        int(obj["ctxs"][names[1]][5]),
    ]

    picks1 = [int(itt.split("=")[-1]) for itt in obj["choices"][names[0]]]
    picks2 = [int(itt.split("=")[-1]) for itt in obj["choices"][names[1]]]

    score1 = compute_score(vals1, picks1)
    score2 = compute_score(vals2, picks2)
    choices = gen_choices(cnts)
    can_improve = False
    for cand1, cand2 in choices:
        cand_score1 = compute_score(vals1, cand1)
        cand_score2 = compute_score(vals2, cand2)
        if (cand_score1 > score1 and cand_score2 >= score2) or (cand_score1 >= score1 and cand_score2 > score2):
            can_improve = True
    
    return not can_improve # if you can improve - you are not optimal.

class BotBotPlay(object):
    """Bot Bot play runner."""
    def __init__(self, dialog, ctx_gen, args, logger=None):
        self.dialog = dialog
        self.ctx_gen = ctx_gen
        self.args = args
        self.logger = logger if logger else DialogLogger()

    def run(self):
        n = 0
        # goes through the list of contexes and kicks off a dialogue
        all_ctxs = []
        for ctxs in self.ctx_gen.iter():
            all_ctxs.append(ctxs)
        
        # all the chats and metrics will be stored here.
        all_chats = []
        for ctxs in tqdm(all_ctxs):
            n += 1
            self.logger.dump('=' * 80)
            _, _, _, obj = self.dialog.run(ctxs, self.logger)

            if obj["agreement_status"] == "agreement":
                obj["pareto_optimal"] = is_pareto_optimal(obj)

            all_chats.append(obj)

            self.logger.dump('=' * 80)
            self.logger.dump('')
            if n % 100 == 0:
                self.logger.dump('%d: %s' % (n, self.dialog.show_metrics()), forced=True)

        return all_chats
    

def get_agent_type(model, smart=False, fast=False):
    if isinstance(model, DialogModel):
        if smart:
            return BatchedRolloutAgent if fast else LstmRolloutAgent
        else:
            return LstmAgent
    else:
        assert False, 'unknown model type: %s' % (model)


def main():
    parser = argparse.ArgumentParser(description='selfplaying script')
    parser.add_argument('--models_dir', type=str,
        help='A directory containing all models.')
    parser.add_argument('--conv_dir', type=str, default='',
        help='directory where we store pairwise interactions.')
    parser.add_argument('--context_file', type=str,
        help='context file')
    parser.add_argument('--temperature', type=float, default=1.0,
        help='temperature')
    parser.add_argument('--verbose', action='store_true', default=False,
        help='print out converations')
    parser.add_argument('--seed', type=int, default=1,
        help='random seed')
    parser.add_argument('--ref_text', type=str,
        help='file with the reference text')
    parser.add_argument('--domain', type=str, default='object_division',
        help='domain for the dialogue')
    args = parser.parse_args()

    utils.set_seed(args.seed)

    #reset the log file
    log_file = os.path.join(args.conv_dir, f"all_logs.log")
    if os.path.exists(log_file):
        os.remove(log_file)

    # sorted list of model names
    model_names = utils.get_model_names(args.models_dir)

    # model pair to a list of chat logs with metrics.
    all_chats = {}

    for ix, mod1 in enumerate(model_names):
        for ij in range(ix, len(model_names)):
            mod2 = model_names[ij]
            key = (mod1, mod2)
            print(key)

            mod1_path = os.path.join(args.models_dir, mod1)
            alice_model = utils.load_model(mod1_path)
            alice_ty = get_agent_type(alice_model)
            alice = alice_ty(alice_model, args, name='Alice')

            mod2_path = os.path.join(args.models_dir, mod2)
            bob_model = utils.load_model(mod2_path)
            bob_ty = get_agent_type(bob_model)
            bob = bob_ty(bob_model, args, name='Bob')

            dialog = Dialog([alice, bob], args)
            
            logger = DialogLogger(verbose=args.verbose, log_file=log_file)
            ctx_gen = DNDContextGenerator(args.context_file)

            bbplay = BotBotPlay(dialog, ctx_gen, args, logger)
            all_chats[key] = bbplay.run() # list of chats will go here.

    out_file = os.path.join(args.conv_dir, f"all_chats.json")
    print(f"All interactions complete. Storing in a json file: {out_file}")
    with open(out_file, "w") as outfile:
        json.dump(all_chats, outfile)


if __name__ == '__main__':
    main()
