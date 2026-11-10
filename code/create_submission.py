"""
Score a solution on the dev set and write its submission.log for the leaderboard.

A solution writes one output directory containing model_def.py, which defines

    def load_model(directory, device):
        ...
        return model   # a torch.nn.Module: token ids [batch, length] (int64) -> logits [batch, length, V]

for any architecture and any weight format. length is at most 2,048, V is at least 50,257, and only the
first 50,257 logits are used. model_def.py may import from this repository's code/ and from its own
directory.

    python3 code/create_submission.py runs/short

It measures the model, then scores it:
- parameters: every distinct tensor in the module's state_dict, at most MAX_PARAMETERS;
- active parameters per token, estimated as the FLOPs of matrix multiplications per token divided by 2
  (each weight a token uses costs one multiply-add); reported, not limited;
- causality: changing later tokens must not change earlier predictions;
- the score: non-overlapping 2,048-token windows of each dev file, mean next-token cross-entropy in nats
  with bf16 autocast; the score is the mean over the four domains.
The last line of submission.log (RESULT, then JSON) is what the leaderboard reads.
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
import data  # noqa: E402

DOMAINS = ["math", "code", "medicine", "history"]
T = 2048  # window length
VOCAB = 50257
MAX_PARAMETERS = 300_000_000


def logits_of(model, x):
    """The model's logits for x, after checking they have the required type and shape."""
    out = model(x)
    if not torch.is_tensor(out) or out.shape[:2] != x.shape or out.shape[-1] < VOCAB:
        got = tuple(out.shape) if torch.is_tensor(out) else type(out).__name__
        raise SystemExit(f"load_model's module must return logits [batch, length, >= {VOCAB}] for every position; "
                         f"for input {tuple(x.shape)} it returned {got}")
    return out


def count_parameters(model):
    """Elements in all distinct tensors of the state_dict (tied weights counted once)."""
    seen = {(t.data_ptr(), t.numel()): t.numel() for t in model.state_dict().values()}
    return sum(seen.values())


@torch.no_grad()
def active_parameters(model, window, device):
    """Weights used per token: matrix-multiply FLOPs per token / 2 (attention scores are not matmul ops here)."""
    with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"), FlopCounterMode(display=False) as counter:
        logits_of(model, window[None])
    flops = counter.get_flop_counts()["Global"]
    return sum(n for op, n in flops.items() if str(op) in ("aten.mm", "aten.addmm")) / 2 / window.numel()


@torch.no_grad()
def causal(model, window, cuts=(512, 1024, 1536), tolerance=1e-3):
    """True if logits before each cut stay the same when every token from the cut on is replaced (fp32)."""
    base = logits_of(model, window[None]).float()
    g = torch.Generator().manual_seed(0)
    for cut in cuts:
        changed = window.clone()
        changed[cut:] = torch.randint(0, VOCAB, (window.numel() - cut,), generator=g).to(window.device)
        if (logits_of(model, changed[None]).float()[:, :cut] - base[:, :cut]).abs().max().item() > tolerance:
            return False
    return True


@torch.no_grad()
def file_loss(model, tokens, batch, device):
    """Mean next-token cross-entropy over the file's non-overlapping T-token windows."""
    n = (len(tokens) - 1) // T
    tokens = torch.from_numpy(tokens[:n * T + 1].astype("int64")).to(device)
    total = 0.0
    for start in range(0, n, batch):
        b = min(batch, n - start)
        chunk = tokens[start * T:(start + b) * T + 1]
        x, y = chunk[:-1].view(b, T), chunk[1:].view(b, T)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = logits_of(model, x)
        total += F.cross_entropy(logits[..., :VOCAB].float().reshape(-1, VOCAB), y.reshape(-1), reduction="sum").item()
    return total / (n * T)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", help="the solution's output directory (with model_def.py)")
    parser.add_argument("--data", default=os.path.join(os.path.dirname(CODE), "dev"),
                        help="directory of <domain>.bin files to score on (default: the dev set)")
    parser.add_argument("--batch", type=int, default=4, help="windows per forward pass")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = os.path.basename(os.path.abspath(args.data))
    lines = []

    def log(text):
        print(text, flush=True)
        lines.append(text)

    sys.path.insert(0, os.path.abspath(args.directory))
    import model_def  # the solution's own model definition
    model = model_def.load_model(args.directory, device).eval()
    window = torch.from_numpy(data.read_tokens(os.path.join(args.data, "math.bin"))[:T].astype("int64")).to(device)
    parameters = count_parameters(model)
    active = active_parameters(model, window, device)
    is_causal = causal(model, window)
    valid = parameters <= MAX_PARAMETERS and is_causal

    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    log(f"submission  {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} | {args.directory} | {gpu} | torch {torch.__version__}")
    log(f"parameters  {parameters:,} (limit {MAX_PARAMETERS:,})")
    log(f"active      about {active / 1e6:.1f}M parameters per token (matrix-multiply FLOPs / 2; not limited)")
    log(f"causal      {'yes' if is_causal else 'NO: later tokens change earlier predictions'}")
    losses = {d: file_loss(model, data.read_tokens(os.path.join(args.data, f"{d}.bin")), args.batch, device) for d in DOMAINS}
    mean = sum(losses.values()) / len(losses)
    log(f"{name:<11} " + " | ".join(f"{d} {v:.4f}" for d, v in losses.items()) + f" | mean {mean:.4f}")
    log("valid" if valid else "INVALID: " + ", ".join(p for p, bad in (("parameters", parameters > MAX_PARAMETERS),
                                                                      ("causality", not is_causal)) if bad))
    log("RESULT " + json.dumps({"data": name, "losses": losses, "mean": mean, "parameters": parameters,
                                "active_parameters": round(active), "causal": is_causal, "valid": valid}))

    out = os.path.join(args.directory, "submission.log" if name == "dev" else f"submission_{name}.log")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
