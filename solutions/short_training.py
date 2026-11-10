"""
Placeholder solution with a short training run, and an example of a changed architecture: the generalist
plus one extra layer (defined in extra_layer_model_def.py), trained with plain next-token loss on an equal
mix of the four domains' pools.

    python3 solutions/short_training.py --out runs/short                       # 200 steps, about 6 minutes on an A10G
    python3 solutions/short_training.py --out runs/short --steps 1000 --lr 2e-4
    python3 code/create_submission.py runs/short

Writes runs/short/model.pt (config and weights), model_def.py (a copy of extra_layer_model_def.py, which
rebuilds the model from model.pt) and train.log.
"""

import argparse
import os
import shutil
import sys
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "code"), os.path.join(ROOT, "solutions")]
import data  # noqa: E402
import gpt  # noqa: E402
from extra_layer_model_def import GPTWithExtraLayer  # noqa: E402  the new architecture

DOMAINS = ["math", "code", "medicine", "history"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--steps", type=int, default=200, help="optimizer steps")
    parser.add_argument("--batch-size", type=int, default=32, help="sequences of 2,048 tokens per step")
    parser.add_argument("--micro-batch", type=int, default=4, help="sequences per forward pass (lower it if out of memory)")
    parser.add_argument("--lr", type=float, default=3e-4, help="peak learning rate")
    parser.add_argument("--warmup", type=int, default=20, help="steps of linear warmup, then linear decay to zero")
    parser.add_argument("--weight-decay", type=float, default=0.3816, help="the provided models' own weight decay")
    parser.add_argument("--log-every", type=int, default=10, help="steps between log lines")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    # ---- 1. Settings ---------------------------------------------------------------------------------
    args = parse_args()
    assert args.batch_size % args.micro_batch == 0, "--batch-size must be a multiple of --micro-batch"
    os.makedirs(args.out, exist_ok=True)
    log_file = open(os.path.join(args.out, "train.log"), "w")

    def log(message):
        print(message, flush=True)
        log_file.write(message + "\n")
        log_file.flush()

    log(f"settings: {vars(args)}")
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 2. Model: the new architecture, with the provided generalist's weights loaded into it --------
    generalist = gpt.load(os.path.join(ROOT, "models", "generalist.bin"))
    model = GPTWithExtraLayer(generalist.config)
    missing, _ = model.load_state_dict(generalist.state_dict(), strict=False)  # every weight but the extra layer
    assert all(name.startswith("extra.") for name in missing), missing
    model = model.to(device).train()
    seq_len = model.config.block_size  # 2,048
    log(f"model: {sum(p.numel() for p in model.parameters()):,} parameters on {device}")

    # ---- 3. Data: each training sequence comes from one of the four pools, chosen uniformly -----------
    pools = [data.read_tokens(os.path.join(ROOT, "pool", f"{d}.bin")) for d in DOMAINS]
    rng = np.random.default_rng(args.seed)

    def get_batch(n):
        """n random windows of seq_len + 1 tokens: inputs, and the next-token targets."""
        rows = []
        for _ in range(n):
            pool = pools[rng.integers(len(pools))]
            start = rng.integers(len(pool) - seq_len - 1)
            rows.append(pool[start:start + seq_len + 1])
        xy = torch.from_numpy(np.stack(rows).astype(np.int64)).to(device)
        return xy[:, :-1], xy[:, 1:]

    # ---- 4. Optimizer: AdamW with the muP groups, linear warmup then linear decay to zero --------------
    optimizer = torch.optim.AdamW(gpt.optimizer_groups(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.95),
                                  eps=1e-8, fused=device == "cuda")

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        return args.lr * (1 - (step - args.warmup) / max(1, args.steps - args.warmup))

    # ---- 5. Train loop ----------------------------------------------------------------------------------
    accum = args.batch_size // args.micro_batch  # forward passes per step
    start = time.time()
    for step in range(args.steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step) * group["lr_mult"]  # muP: hidden matrices get lr / width_mult
        step_loss = 0.0
        for _ in range(accum):
            x, y = get_batch(args.micro_batch)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(x)
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1))
            (loss / accum).backward()
            step_loss += loss.item() / accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            log(f"step {step + 1}/{args.steps} | loss {step_loss:.4f} | lr {lr_at(step):.2e} | {time.time() - start:.0f}s")

    # ---- 6. Save: the weights, and the model definition as model_def.py ---------------------------------
    torch.save({"config": asdict(model.config), "state_dict": model.state_dict()}, os.path.join(args.out, "model.pt"))
    shutil.copy(os.path.join(ROOT, "solutions", "extra_layer_model_def.py"), os.path.join(args.out, "model_def.py"))
    log(f"saved {args.out}/model.pt and model_def.py ({time.time() - start:.0f}s of training)")


if __name__ == "__main__":
    main()
