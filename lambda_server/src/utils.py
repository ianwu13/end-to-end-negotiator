import os
import random
import numpy as np
import torch
from models.dialog_models import DialogModel
import data
from agent import LstmAgent
from torch.autograd import Variable
import torch.nn.functional as F


def use_cuda(enabled, device_id=0):
    """Verifies if CUDA is available and sets default device to be device_id."""
    if not enabled:
        return None
    assert torch.cuda.is_available(), 'CUDA is not available'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(device_id)
    return device_id

class ArgsClass:
    def __init__(self) -> None:
        self.temperature = 0.5
        self.domain = "object_division"

def load_model(mpath):
    """
    Load model from mpath.
    """
    if torch.cuda.is_available():
        checkpoint = torch.load(mpath)
    else:
        checkpoint = torch.load(mpath, map_location=torch.device("cpu"))

    model_args = checkpoint["args"]

    device_id = use_cuda(model_args.cuda)
    corpus = data.WordCorpus(model_args.data, freq_cutoff=model_args.unk_threshold, verbose=False)
    model = DialogModel(corpus.word_dict, corpus.item_dict, corpus.context_dict,
        corpus.output_length, model_args, device_id)

    model.load_state_dict(checkpoint['state_dict'])

    args = ArgsClass()
    final_model_obj = LstmAgent(model, args, name='Alice')

    return final_model_obj


def load_models():
    """
    Load models.
    """
    mod_names = [
        "sv_model.pt",
        "rl_model_rw_utility_1_0_0_0.pt",
        "rl_model_rw_utility_1_0_-0.75_-0.75.pt",
        "rl_selfish_ag_fair_rw_own_points.pt",
        "rl_selfish_ag_selfish_rw_own_points.pt",
        "rl_fair_ag_fair_rw_fair.pt",
        "rl_fair_ag_selfish_rw_fair.pt",
    ]

    name2mod = {}
    for mod_name in mod_names:
        mod_path = os.path.join("trained_ckpts", mod_name)
        mod = load_model(mod_path)
        name2mod[mod_name] = mod
    
    return name2mod


class DNDContextGenerator(object):
    """Dialogue context generator. Generates contexes from the file in standard DND format - basically to be used for extracting the contexts directly from the data/negotiate/test.txt file for bot_bot play."""
    def __init__(self, context_file):
        ctxs = []
        with open(context_file, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                ctx_pair = [data.get_tag(tokens, "input"), data.get_tag(tokens, "partner_input")]
                ctxs.append(ctx_pair)
        print(f"ctxs: {len(ctxs)}")

        #only keep the unique ones
        cset = set()
        ctxs2 = []
        for item in ctxs:
            if " ".join(item[0] + item[1]) in cset:
                continue
            ctxs2.append(item)
            cset.add(" ".join(item[0] + item[1]))
        print(f"ctxs2: {len(ctxs2)}")

        # remove bad_ixs
        bad_ixs = set()
        for ix, item in enumerate(ctxs2):
            f = 0
            for item2 in ctxs2:
                if item[::-1] == item2:
                    f = 1
                    break
            if not f:
                # this is bad and should be removed.
                bad_ixs.add(ix)

        # filter out bad ones.
        ctxs3 = []
        for ix, item in enumerate(ctxs2):
            if ix in bad_ixs:
                continue
            ctxs3.append(item)

        print(f"ctxs3: {len(ctxs3)}")

        self.ctxs = ctxs3[:]

        #validate

        #no bad ones
        for item in self.ctxs:
            f = 0
            for item2 in self.ctxs:
                if item[::-1] == item2:
                    f = 1
                    break
            assert f, item

        #uniqueness
        cset = set()
        for item in self.ctxs:
            cset.add(" ".join(item[0] + item[1]))
        assert len(cset) == len(self.ctxs)

        print(f"Num ctx pairs loaded: {len(self.ctxs)}")

    def sample(self):
        return random.choice(self.ctxs)

    def iter(self, nepoch=1):
        for e in range(nepoch):
            # random.shuffle(self.ctxs) # no randomization required since this class is only for bot-bot evaluation via the bot_bot_play script.
            for ctx in self.ctxs:
                yield ctx           


def load_context_pairs():
    """
    Load context pairs.
    """
    ctx_gen = DNDContextGenerator("data/negotiate/test.txt")
    # goes through the list of contexes and returns a list of strings
    all_ctxs = []
    for ctxs in ctx_gen.iter():
        ctxs_str = " ".join(ctxs[0] + ctxs[1])
        all_ctxs.append(ctxs_str)
    return all_ctxs


def feed_context(model_obj, agent_cxt):
    """
    Feed context to model.
    """
    # the hidden state of all the pronounced words
    lang_hs = []
    # all the pronounced words
    words = []
    # encoded context
    ctx = model_obj._encode(agent_cxt, model_obj.model.context_dict)
    # hidded state of context
    ctx_h = model_obj.model.forward_context(Variable(ctx))
    # current hidden state of the language rnn
    lang_h = model_obj.model.zero_hid(1)

    return lang_h, ctx_h, lang_hs, words


def read(model_obj, lang_h, ctx_h, lang_hs, words, inpt):
    """
    Read new human utterance from the payload.

    inpt: tokens of the human utterance.
    """
    inpt = model_obj._encode(inpt, model_obj.model.word_dict)
    curr_lang_hs, lang_h = model_obj.model.read(Variable(inpt), lang_h, ctx_h)
    # append new hidded states to the current list of the hidden states
    lang_hs.append(curr_lang_hs.squeeze(1))
    # first add the special 'THEM:' token
    words.append(model_obj.model.word2var('THEM:').unsqueeze(1))
    # then read the utterance
    words.append(Variable(inpt))
    assert (torch.cat(words).size()[0] == torch.cat(lang_hs).size()[0])

    return lang_h, ctx_h, lang_hs, words


def write(model_obj, lang_h, ctx_h, lang_hs, words):
    """
    Write new agent utterance.
    """
    # generate a new utterance
    _, outs, lang_h, curr_lang_hs = model_obj.model.write(lang_h, ctx_h,
        100, model_obj.args.temperature)
    # append new hidded states to the current list of the hidden states
    lang_hs.append(curr_lang_hs)
    # first add the special 'YOU:' token
    words.append(model_obj.model.word2var('YOU:').unsqueeze(1))
    # then append the utterance
    words.append(outs)
    assert (torch.cat(words).size()[0] == torch.cat(lang_hs).size()[0])
    # decode into English words
    resp_tokens = model_obj._decode(outs, model_obj.model.word_dict)

    return resp_tokens, lang_h, ctx_h, lang_hs, words


def make_choice(model_obj, lang_h, ctx_h, lang_hs, words, agent_cxt):
    """
    Make the final choice - either a deal or walk away.
    """
    # get all the possible choices
    choices = model_obj.domain.generate_choices(agent_cxt)
    # concatenate the list of the hidden states into one tensor
    curr_lang_hs = torch.cat(lang_hs)
    # concatenate all the words into one tensor
    curr_words = torch.cat(words)
    # logits for each of the item
    logits = model_obj.model.generate_choice_logits(curr_words, curr_lang_hs, ctx_h)

    # construct probability distribution over only the valid choices
    choices_logits = []
    for i in range(model_obj.domain.selection_length()):
        idxs = [model_obj.model.item_dict.get_idx(c[i]) for c in choices]
        idxs = Variable(torch.from_numpy(np.array(idxs)))
        idxs = model_obj.model.to_device(idxs)
        choices_logits.append(torch.gather(logits[i], 0, idxs).unsqueeze(1))

    choice_logit = torch.sum(torch.cat(choices_logits, 1), 1, keepdim=False)
    # subtract the max to softmax more stable
    choice_logit = choice_logit.sub(choice_logit.max().item())

    # http://pytorch.apachecn.org/en/0.3.0/_modules/torch/nn/functional.html
    # choice_logit.dim() == 1, so implicitly _get_softmax_dim returns 0
    prob = F.softmax(choice_logit,dim=0)
    # take the most probably choice
    _, idx = prob.max(0, keepdim=True)

    # Pick only your choice
    return choices[idx.item()][:model_obj.domain.selection_length()]


def make_unsafe(utt):
    """
    $ -> <
    # -> >
    """
    utt = utt.replace("$", "<")
    utt = utt.replace("#", ">")
    return utt


def make_safe(utt):
    """
    < -> $
    > -> #
    """
    utt = utt.replace("<", "$")
    utt = utt.replace(">", "#")
    return utt


def get_model_response(payload, model_obj, lioness_obj):
    """
    Get model response.
    payload: input payload from the UI
    model_obj: current model object for the user
    lioness_obj: current lioness storage object for the user

    model states to remember:
     - lang_h, ctx_h, lang_hs, words
     - conv

    """
    
    # agent cxt
    agent_cxt = payload["cxt"].split()[6:] # assumed order is (human, agent)

    if not lioness_obj:
        # feed in the context
        lang_h, ctx_h, lang_hs, words = feed_context(model_obj, agent_cxt)
        conv = []
    else:
        # get the lioness states
        lang_h, ctx_h, lang_hs, words, conv = lioness_obj["lang_h"], lioness_obj["ctx_h"], lioness_obj["lang_hs"], lioness_obj["words"], lioness_obj["conv"]

    if payload["human_utt"]:
        # the human has said something new; read it before writing
        utt_tokens = make_unsafe(payload["human_utt"]).strip().lower().split()
        lang_h, ctx_h, lang_hs, words = read(model_obj, lang_h, ctx_h, lang_hs, words, utt_tokens)
        utt_obj = {
            "name": "human",
            "sent": " ".join(utt_tokens),
        }
        conv.append(utt_obj)
    
    # see if the human has already outputted a selection token
    if conv and conv[-1]["name"] == "human" and "<selection>" in conv[-1]["sent"]:
        # agent response is just the selection token
        resp = "<selection>"
    else:
        # no sign of conv being over; get the model response
        resp_tokens, lang_h, ctx_h, lang_hs, words = write(model_obj, lang_h, ctx_h, lang_hs, words)
        resp = " ".join(resp_tokens)
    
    # add the agent response to the conversation
    utt_obj = {
        "name": "agent",
        "sent": resp,
    }
    conv.append(utt_obj)

    if "<selection>" in conv[-1]:
        agent_chose = make_choice(model_obj, lang_h, ctx_h, lang_hs, words, agent_cxt)

    # prepare outputs
    out_resp_obj = {
        "resp": make_safe(resp),
        "meta": agent_chose
    }

    out_lioness_obj = {
        "lang_h": lang_h,
        "ctx_h": ctx_h,
        "lang_hs": lang_hs,
        "words": words,
        "conv": conv
    }

    return out_resp_obj, out_lioness_obj





    

    

