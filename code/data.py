"""
Token files. The pool, the dev set and the hidden test are all in one format: a 1,024-byte header of 256
int32 values (magic 20240520, version, token count), then the GPT-2 token ids as uint16. Documents are
separated by the end-of-text token (50256), placed before each document. tiktoken's "gpt2" encoding
decodes the ids back to text.

    tokens = data.read_tokens("pool/math.bin")   # numpy uint16 array
"""

import numpy as np

TOKENS_MAGIC = 20240520
EOT = 50256


def read_tokens(path):
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == TOKENS_MAGIC, f"{path}: not a token file"
    tokens = np.fromfile(path, dtype=np.uint16, offset=1024)
    assert tokens.size == header[2], f"{path}: header says {header[2]} tokens, found {tokens.size}"
    return tokens


def write_tokens(path, tokens):
    header = np.zeros(256, dtype=np.int32)
    header[:3] = TOKENS_MAGIC, 1, len(tokens)
    with open(path, "wb") as f:
        f.write(header.tobytes() + np.asarray(tokens, dtype=np.uint16).tobytes())
