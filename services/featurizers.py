"""
services/featurizers.py
Deterministic, dependency-free featurizers that turn real inputs into model-ready vectors.

- SMILES  -> hashed atom-token n-gram counts (used by the EGFR affinity model and the fusion chemistry tokens)
- Text    -> signed hashed unigram/bigram embedding (used for the clinical tokens in fusion)

Hashing uses zlib.crc32 so features are identical across runs, machines and Python versions.
"""

import re
import zlib
from typing import List

import numpy as np

# Two-letter elements and bracket atoms must stay one token, otherwise "Cl" would read as C + l.
_SMILES_TOKEN = re.compile(r"(\[[^\]]+\]|Br|Cl|Si|Se|se|@@|%\d{2}|[A-Za-z]|\d|\(|\)|=|#|\+|-|/|\\|\.|@|\*|:|~)")


def tokenize_smiles(smiles: str) -> List[str]:
    return _SMILES_TOKEN.findall(smiles)


def _bucket(token: str, dim: int) -> int:
    return zlib.crc32(token.encode("utf-8")) % dim


def smiles_features(smiles: str, dim: int = 2048, max_n: int = 4) -> np.ndarray:
    """Counts of SMILES token n-grams (n = 1..max_n) hashed into `dim` buckets, log-scaled and L2-normalised."""
    tokens = tokenize_smiles(smiles)
    vec = np.zeros(dim, dtype=np.float64)
    for n in range(1, max_n + 1):
        for i in range(len(tokens) - n + 1):
            vec[_bucket(f"{n}:" + "".join(tokens[i:i + n]), dim)] += 1.0
    vec = np.log1p(vec)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def text_embedding(text: str, dim: int = 768) -> np.ndarray:
    """Signed feature hashing of lower-cased word unigrams and bigrams, L2-normalised."""
    words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
    grams = words + [f"{a} {b}" for a, b in zip(words, words[1:])]
    vec = np.zeros(dim, dtype=np.float64)
    for g in grams:
        h = zlib.crc32(g.encode("utf-8"))
        vec[h % dim] += 1.0 if (h >> 16) & 1 else -1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def text_tokens(sentences: List[str], dim: int = 768) -> np.ndarray:
    """One embedding row per sentence, so the clinical context becomes a short token sequence."""
    return np.stack([text_embedding(s, dim) for s in sentences if s.strip()])
