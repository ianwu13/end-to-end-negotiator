"""
Use saved logs and compute pair-wise and overall results, build any plots.

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


def process_convs(fname):
    """Create a list of convs."""
    pass


def get_key(fname):
    items = fname.split(".pt_")
    key = (items[0], items[1].replace(".pt.log",""))
    return key


def main():
    parser = argparse.ArgumentParser(description='selfplaying script')
    
    parser.add_argument('--conv_dir', type=str, default='',
        help='directory where we stored pairwise interactions.')
    parser.add_argument('--results_dir', type=str, default='',
        help='directory where we store the results, plots, etc.')
    
    args = parser.parse_args()

    # sorted list of pairwise conversation fnames
    pw_conv_fnames = utils.get_pw_conv_fnames(args.conv_dir)

    # pair tuple (sorted) to results for that pair..basically all conversations should be stored in this way.
    results = {}

    for ix, fname in enumerate(pw_conv_fnames):
        print(f"{ix}/{len(pw_conv_fnames)}: {fname}")

        key = get_key(fname)
        results[key] = process_convs(fname)

    out_path = os.path.join(args.results_dir, "everything1.json")
    with open(out_path, "w") as outfile:
        json.dump(results, outfile)

if __name__ == '__main__':
    main()
