from __future__ import annotations

import os
import random

import numpy as np

from src.config import SEED


def set_seed(seed: int = SEED) -> int:
    """Фиксирует seed для random, numpy и хеширования строк

    Возвращает использованный seed, чтобы его можно было записать в лог прогона.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed
