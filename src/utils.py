# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Various helpers.
"""

import random
import copy
import pdb
import sys

import torch
import numpy as np
from models.dialog_model import DialogModel
import data
import os


def backward_hook(grad):
    """Hook for backward pass."""
    print(grad)
    pdb.set_trace()
    return grad


def save_model(model, file_name):
    """Serializes model to a file."""
    torch.save({'state_dict': model.state_dict(), 'args': model.args}, file_name)

def load_model(file_name):
    """Reads model from a file."""

    if torch.cuda.is_available():
        checkpoint = torch.load(file_name)
    else:
        checkpoint = torch.load(file_name, map_location=torch.device("cpu"))

    model_args = checkpoint["args"]

    device_id = use_cuda(model_args.cuda)
    corpus = data.WordCorpus(model_args.data, freq_cutoff=model_args.unk_threshold, verbose=False)
    model = DialogModel(corpus.word_dict, corpus.item_dict, corpus.context_dict,
        corpus.output_length, model_args, device_id)

    model.load_state_dict(checkpoint['state_dict'])
    return model

def set_seed(seed):
    """Sets random seed everywhere."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def use_cuda(enabled, device_id=0):
    """Verifies if CUDA is available and sets default device to be device_id."""
    if not enabled:
        return None
    assert torch.cuda.is_available(), 'CUDA is not available'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(device_id)
    return device_id


def prob_random():
    """Prints out the states of various RNGs."""
    print('random state: python %.3f torch %.3f numpy %.3f' % (
        random.random(), torch.rand(1)[0], np.random.rand()))
    

def get_model_names(models_dir):
    """Get all model names inside this dir."""
    model_names = []

    for item in os.listdir(models_dir):
        if os.path.isfile(os.path.join(models_dir, item)):
            assert "rl_model" in item or "sv_model" in item
            assert ".pt" in item
            model_names.append(item)
    
    return model_names


class ContextGenerator(object):
    """Dialogue context generator. Generates contexes from the file."""
    def __init__(self, context_file):
        self.ctxs = []
        with open(context_file, 'r') as f:
            ctx_pair = []
            for line in f:
                ctx = line.strip().split()
                ctx_pair.append(ctx)
                if len(ctx_pair) == 2:
                    self.ctxs.append(ctx_pair)
                    ctx_pair = []

    def sample(self):
        return random.choice(self.ctxs)

    def iter(self, nepoch=1):
        for e in range(nepoch):
            random.shuffle(self.ctxs)
            for ctx in self.ctxs:
                yield ctx


class DNDContextGenerator(object):
    """Dialogue context generator. Generates contexes from the file in standard DND format - basically to be used for extracting the contexts directly from the data/negotiate/test.txt file."""
    def __init__(self, context_file):
        self.ctxs = []
        with open(context_file, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                ctx_pair = [data.get_tag(tokens, "input"), data.get_tag(tokens, "partner_input")]
                self.ctxs.append(ctx_pair)

        print(f"Num ctx pairs loaded: {len(self.ctxs)}")

        # validate
        for ix, item in enumerate(self.ctxs):
            f = 0
            for ij, item2 in enumerate(self.ctxs):
                if ij != ix and item[::-1] == item2:
                    f = 1
                    break
            assert f, f"{ix}: {item}"

    def sample(self):
        return random.choice(self.ctxs)

    def iter(self, nepoch=1):
        for e in range(nepoch):
            random.shuffle(self.ctxs)
            for ctx in self.ctxs:
                yield ctx


class ManualContextGenerator(object):
    """Dialogue context generator. Takes contexes from stdin."""
    def __init__(self, num_types=3, num_objects=10, max_score=10):
            self.num_types = num_types
            self.num_objects = num_objects
            self.max_score = max_score

    def _input_ctx(self):
        while True:
            try:
                ctx = input('Input context: ')
                ctx = ctx.strip().split()
                if len(ctx) != 2 * self.num_types:
                    raise
                if np.sum([int(x) for x in ctx[0::2]]) != self.num_objects:
                    raise
                if np.max([int(x) for x in ctx[1::2]]) > self.max_score:
                    raise
                return ctx
            except KeyboardInterrupt:
                sys.exit()
            except:
                print('The context is invalid! Try again.')
                print('Reason: num_types=%d, num_objects=%d, max_score=%s' % (
                    self.num_types, self.num_objects, self.max_score))

    def _update_scores(self, ctx):
        for i in range(1, len(ctx), 2):
            ctx[i] = np.random.randint(0, self.args.max_score + 1)
        return ctx

    def sample(self):
        ctx1 = self._input_ctx()
        ctx2 = self._update_scores(copy.copy(ctx1))
        return [ctx1, ctx2]
