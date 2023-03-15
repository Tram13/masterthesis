import numpy as np
from pathlib import Path


def load_embeddings(path: Path):
    return np.load(path.with_suffix('.npy'))