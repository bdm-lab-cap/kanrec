"""
Fijación completa de semillas.

`torch.manual_seed()` por sí solo NO hace reproducible un experimento: no
cubre el generador de NumPy (que usa el muestreo de datos), el de la
biblioteca estándar, ni el de CUDA (inicialización de pesos y dropout en
GPU). El proyecto declaraba reproducibilidad fijando solo el primero, lo que
la hacía parcial.

`set_seed()` fija los cuatro y, opcionalmente, activa el modo determinista de
cuDNN. Ese modo se deja desactivado por defecto porque puede reducir el
rendimiento entre un 10 % y un 30 %: para comparar encoders basta con que
todos partan del mismo estado, que es lo que garantizan las semillas.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Fija todos los generadores de números aleatorios en juego.

    Args:
        seed:          semilla a aplicar.
        deterministic: si True, fuerza además algoritmos deterministas en
                       cuDNN. Cuesta rendimiento; úsese solo cuando se
                       requiera reproducibilidad bit a bit en GPU.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)      # no-op si no hay GPU
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
