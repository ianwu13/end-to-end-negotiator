import os
import random
import torch
from models.dialog_model import DialogModel
import data
from agent import LstmAgent


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
    ctx_gen = DNDContextGenerator("cxts/test.txt")
    # goes through the list of contexes and kicks off a dialogue
    all_ctxs = []
    for ctxs in ctx_gen.iter():
        all_ctxs.append(ctxs)
    return all_ctxs


def get_model_response(payload, storage):
    """
    Get model response.
    payload: new payload from UI
    storage: current lioness storage object for the user
    """
    pass
