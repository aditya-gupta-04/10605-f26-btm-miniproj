"""
model_def.py for a solution whose model is one checkpoint in the released format (model.bin): the
placeholder solutions copy this file into their output directory. create_submission.py calls
load_model(directory, device) and needs back a module mapping token ids to logits.
"""

import os

import gpt  # code/gpt.py; create_submission.py puts code/ on the import path


def load_model(directory, device):
    return gpt.load(os.path.join(directory, "model.bin"), device)
