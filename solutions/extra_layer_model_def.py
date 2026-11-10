"""
model_def.py of the short-training placeholder: the provided GPT plus one extra layer.

create_submission.py needs one thing from a model_def.py: load_model(directory, device), returning a
torch.nn.Module that maps token ids [batch, length] to logits [batch, length, 50257]. This model isn't
the provided architecture, so the file also defines its class, which load_model needs to rebuild it.
short_training.py imports the class to train the model, saves model.pt, and copies this file into its
output directory as model_def.py.
"""

import os

import torch
import torch.nn as nn

import gpt  # code/gpt.py; create_submission.py puts code/ on the import path


class GPTWithExtraLayer(gpt.GPT):
    """The provided GPT plus a residual MLP (768 -> 256 -> 768) after the last block: 0.4M more parameters."""

    def __init__(self, config):
        super().__init__(config)
        self.extra = nn.Sequential(nn.LayerNorm(config.n_embd), nn.Linear(config.n_embd, 256),
                                   nn.GELU(approximate="tanh"), nn.Linear(256, config.n_embd))
        nn.init.zeros_(self.extra[-1].weight)  # adds nothing at first: the model starts as the GPT it is built from
        nn.init.zeros_(self.extra[-1].bias)

    def forward(self, idx):
        t = self.transformer
        x = t.wte(idx) + t.wpe(torch.arange(idx.shape[1], device=idx.device))
        for block in t.h:
            x = block(x)
        x = x + self.extra(x)
        return self.lm_head(t.ln_f(x) / self.config.width_mult)


def load_model(directory, device):
    saved = torch.load(os.path.join(directory, "model.pt"), map_location="cpu", weights_only=True)
    model = GPTWithExtraLayer(gpt.Config(**saved["config"]))
    model.load_state_dict(saved["state_dict"])
    return model.to(device)
