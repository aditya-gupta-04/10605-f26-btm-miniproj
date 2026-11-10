"""
Download the release's models and training pool from Hugging Face into this repository, then check every
file against MANIFEST.sha256. Run it once, from anywhere, after cloning:

    python3 code/download_data.py

It fetches models/ (the four experts, the generalist and the seed: 6 x 250 MB) and pool/ (one 120 MB
file per domain), pinned to one revision so everyone gets the same files. The code and the dev set
come with the repository itself.
"""

import hashlib
import os

from huggingface_hub import snapshot_download

REPO = "adityagupta04/btm-miniproj-f26"  # the release on Hugging Face
REVISION = "release-v0"  # a fixed tag
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main():
    snapshot_download(REPO, revision=REVISION, repo_type="dataset", local_dir=ROOT,
                      allow_patterns=["models/*", "pool/*"])
    bad = []
    for line in open(os.path.join(ROOT, "MANIFEST.sha256")):
        digest, name = line.split(maxsplit=1)
        path = os.path.join(ROOT, name.strip())
        if not os.path.exists(path) or sha256(path) != digest:
            bad.append(name.strip())
    if bad:
        raise SystemExit(f"{len(bad)} files are missing or differ from MANIFEST.sha256: {bad}")
    print("release ok: models/, pool/, dev/ and code/ all match MANIFEST.sha256")


if __name__ == "__main__":
    main()
