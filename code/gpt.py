"""
The released models' architecture and checkpoint format. Every released checkpoint (the seed, the
generalist and the four experts) is a GPT-2 small, 124M parameters at context 2,048, trained with the
maximal update parametrization (muP), stored as a binary file: a 1,024-byte header of 256 int32 values,
then every tensor in a fixed order.

    model = gpt.load("models/generalist.bin", "cuda")   # a GPT module in eval mode
    logits = model(idx)                                 # idx [batch, length] -> [batch, length, 50257]
    gpt.save(model, "runs/mine/model.bin", gpt.read_header("models/seed.bin"))

What muP changes from plain GPT-2:
- attention scores are scaled by attn_mult / head_size instead of 1 / sqrt(head_size);
- the final hidden state is divided by width_mult before the output layer (tied to the token embedding);
- training: hidden matrices take lr / width_mult and weight decay x width_mult (optimizer_groups).
The header stores width_mult and attn_mult, so load() rebuilds the model exactly. Keep the header when
saving: a checkpoint saved without its muP fields loads as plain GPT-2 and scores about 9.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_MAGIC = 20240326
MUP_MAGIC = 0x4D555031  # "MUP1"
VERSIONS = {3: torch.float32, 5: torch.bfloat16}  # header[1]: how the tensors are stored
VOCAB = 50257  # GPT-2's vocabulary; the file pads the embedding to header[7] rows


@dataclass
class Config:
    block_size: int = 2048
    vocab_size: int = VOCAB
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    width_mult: float = 1.0  # muP: n_embd / reference width
    attn_mult: float = 1.0   # muP: attention scale numerator


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.scale = config.attn_mult / (config.n_embd // config.n_head)  # muP; plain GPT-2 uses 1/sqrt

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                   for t in self.c_attn(x).split(C, dim=2))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
        return self.c_proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f=nn.LayerNorm(config.n_embd)))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight  # tied

    def forward(self, idx):
        t = self.transformer
        x = t.wte(idx) + t.wpe(torch.arange(idx.shape[1], device=idx.device))
        for block in t.h:
            x = block(x)
        return self.lm_head(t.ln_f(x) / self.config.width_mult)


def read_header(path):
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == MODEL_MAGIC, f"{path}: not a checkpoint in the released format"
    assert header[1] in VERSIONS, f"{path}: unsupported version {header[1]}"
    return header


def config_from_header(header):
    use_mup = header[8] == MUP_MAGIC and header[10]
    width_mult, attn_mult = header[11:13].view(np.float32) if use_mup else (1.0, np.sqrt(header[6] / header[5]))
    return Config(block_size=int(header[2]), vocab_size=int(header[3]), n_layer=int(header[4]),
                  n_head=int(header[5]), n_embd=int(header[6]), width_mult=float(width_mult),
                  attn_mult=float(attn_mult))


def tensor_shapes(header):
    """(name, shape) of every stored tensor, in file order. The embedding has header[7] (padded) rows."""
    maxT, L, C, Vp = int(header[2]), int(header[4]), int(header[6]), int(header[7])
    per_layer = [("ln_1.weight", (C,)), ("ln_1.bias", (C,)),
                 ("attn.c_attn.weight", (3 * C, C)), ("attn.c_attn.bias", (3 * C,)),
                 ("attn.c_proj.weight", (C, C)), ("attn.c_proj.bias", (C,)),
                 ("ln_2.weight", (C,)), ("ln_2.bias", (C,)),
                 ("mlp.c_fc.weight", (4 * C, C)), ("mlp.c_fc.bias", (4 * C,)),
                 ("mlp.c_proj.weight", (C, 4 * C)), ("mlp.c_proj.bias", (C,))]
    shapes = [("transformer.wte.weight", (Vp, C)), ("transformer.wpe.weight", (maxT, C))]
    for suffix, shape in per_layer:  # all layers of one tensor kind, then the next kind
        shapes += [(f"transformer.h.{i}.{suffix}", shape) for i in range(L)]
    return shapes + [("transformer.ln_f.weight", (C,)), ("transformer.ln_f.bias", (C,))]


def read_checkpoint(path):
    """(header, {name: float32 tensor}), with the embedding's padded rows as stored."""
    header = read_header(path)
    shapes = tensor_shapes(header)
    n = sum(int(np.prod(s)) for _, s in shapes)
    if VERSIONS[header[1]] == torch.bfloat16:  # widen to float32 exactly
        raw = np.fromfile(path, dtype=np.uint16, offset=1024, count=n)
        flat = (raw.astype(np.uint32) << 16).view(np.float32)
    else:
        flat = np.fromfile(path, dtype=np.float32, offset=1024, count=n)
    assert flat.size == n, f"{path}: truncated"
    params, i = {}, 0
    for name, shape in shapes:
        size = int(np.prod(shape))
        params[name] = torch.from_numpy(flat[i:i + size].reshape(shape).copy())
        i += size
    return header, params


def write_checkpoint(path, header, params, dtype=torch.bfloat16):
    """Write {name: tensor} under a copy of a released header (which keeps the muP fields). The embedding
    may have 50,257 rows or the padded count; missing padding rows are written as zeros."""
    header = header.copy()
    header[1] = {v: k for k, v in VERSIONS.items()}[dtype]
    with open(path, "wb") as f:
        f.write(header.astype(np.int32).tobytes())
        for name, shape in tensor_shapes(header):
            t = params[name].detach().float().cpu()
            if name == "transformer.wte.weight" and t.shape[0] < shape[0]:
                t = torch.cat([t, torch.zeros(shape[0] - t.shape[0], shape[1])])
            assert tuple(t.shape) == shape, f"{name}: shape {tuple(t.shape)}, expected {shape}"
            f.write(t.to(dtype).view(torch.int16 if dtype == torch.bfloat16 else torch.int32).numpy().tobytes())


def load(path, device="cpu"):
    """A released-format checkpoint as a GPT module, in eval mode."""
    header, params = read_checkpoint(path)
    model = GPT(config_from_header(header))
    with torch.no_grad():
        for name, p in model.named_parameters():  # lm_head.weight is tied to wte and not listed
            p.copy_(params[name][:p.shape[0]])
    return model.to(device).eval()


def save(model, path, header):
    """Save a GPT module in the released format, under `header` (for example read_header(SEED))."""
    write_checkpoint(path, header, dict(model.named_parameters()))


def optimizer_groups(model, weight_decay):
    """AdamW groups for muP training. Each has an "lr_mult" to multiply the learning rate by:
    hidden matrices lr / width_mult with decay x width_mult; embeddings plain; vectors undecayed."""
    w = model.config.width_mult
    groups = {"hidden": ([], 1 / w, weight_decay * w), "embedding": ([], 1.0, weight_decay), "vector": ([], 1.0, 0.0)}
    for name, p in model.named_parameters():
        kind = "vector" if p.dim() < 2 else "embedding" if name.startswith(("transformer.wte", "transformer.wpe")) else "hidden"
        groups[kind][0].append(p)
    return [{"params": ps, "lr_mult": m, "weight_decay": wd} for ps, m, wd in groups.values() if ps]
