"""
Placeholder solution without training: the uniform average of the four experts' weights.

    python3 solutions/average_experts.py --out runs/average
    python3 code/create_submission.py runs/average

Writes runs/average/model.bin (a checkpoint in the provided format), model_def.py (a copy of
gpt_model_def.py, which loads it) and train.log.
"""

import argparse
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
import gpt  # noqa: E402

DOMAINS = ["math", "code", "medicine", "history"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="output directory")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    log_file = open(os.path.join(args.out, "train.log"), "w")

    def log(message):
        print(message, flush=True)
        log_file.write(message + "\n")

    # ---- 1. Load the four experts: each a header and a dict of tensors -------------------------------
    experts = [gpt.read_checkpoint(os.path.join(ROOT, "models", f"expert_{d}.bin")) for d in DOMAINS]
    header = experts[0][0]
    log(f"loaded {len(experts)} experts")

    # ---- 2. Average every tensor -------------------------------------------------------------------------
    average = {name: sum(params[name] for _, params in experts) / len(experts) for name in experts[0][1]}

    # ---- 3. Save in the provided format (the header keeps the muP fields), and the model definition -----
    gpt.write_checkpoint(os.path.join(args.out, "model.bin"), header, average)
    shutil.copy(os.path.join(ROOT, "solutions", "gpt_model_def.py"), os.path.join(args.out, "model_def.py"))
    log(f"saved {args.out}/model.bin and model_def.py")


if __name__ == "__main__":
    main()
