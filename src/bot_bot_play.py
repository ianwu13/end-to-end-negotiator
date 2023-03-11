# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Enable Bot-Bot play in an nC2 manner and save all the logs.
"""

import argparse
import os
from tqdm import tqdm

from agent import LstmAgent, LstmRolloutAgent, BatchedRolloutAgent
import utils
from utils import DNDContextGenerator
from dialog import Dialog, DialogLogger
from models.dialog_model import DialogModel


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
        
        for ctxs in tqdm(all_ctxs):
            n += 1
            self.logger.dump('=' * 80)
            self.dialog.run(ctxs, self.logger)
            self.logger.dump('=' * 80)
            self.logger.dump('')
            if n % 100 == 0:
                self.logger.dump('%d: %s' % (n, self.dialog.show_metrics()), forced=True)


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

    # sorted list of model names
    model_names = utils.get_model_names(args.models_dir)

    for ix, mod1 in enumerate(model_names):
        for ij in range(ix, len(model_names)):
            mod2 = model_names[ij]
            print(mod1, mod2)

            mod1_path = os.path.join(args.models_dir, mod1)
            alice_model = utils.load_model(mod1_path)
            alice_ty = get_agent_type(alice_model)
            alice = alice_ty(alice_model, args, name='Alice')

            mod2_path = os.path.join(args.models_dir, mod2)
            bob_model = utils.load_model(mod2_path)
            bob_ty = get_agent_type(bob_model)
            bob = bob_ty(bob_model, args, name='Bob')

            dialog = Dialog([alice, bob], args)
            log_file = os.path.join(args.conv_dir, f"{mod1}_{mod2}.log")
            logger = DialogLogger(verbose=args.verbose, log_file=log_file)
            ctx_gen = DNDContextGenerator(args.context_file)

            bbplay = BotBotPlay(dialog, ctx_gen, args, logger)
            bbplay.run()


if __name__ == '__main__':
    main()
