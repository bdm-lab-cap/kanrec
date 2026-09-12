"""
Encoder KAN vectorizado.

Motivación (medición, no intuición)
-----------------------------------
El perfilado de latencia sobre GPU T4 mostró que el encoder numérico consume
el **84,5 %** del tiempo de inferencia del modelo completo, y que KAN-REC era
4,05 veces más lento que la normalización directa (10,45 ms frente a 2,58 ms por
batch de 4096).

La causa no es que evaluar B-splines sea caro en sí, sino que
`KANNumericalEncoder.forward` recorre los campos en un **bucle de Python**:

    for j, kan in enumerate(self.field_kans):
        ej = kan(x[:, j:j+1])          # 13 lanzamientos de kernel por batch

En GPU, 13 kernels pequeños y secuenciales desperdician el paralelismo: cada
lanzamiento tiene un coste fijo que domina cuando el trabajo por kernel es
diminuto (un campo, una columna).

Cómo se vectoriza
-----------------
No sirve un único `KANLinear(in_features=13)`: su `F.linear` final mezcla los
13 campos entre sí, y aquí hacen falta 13 funciones **independientes**
φⱼ: R -> R^d, no una función de 13 variables.

La observación que lo hace posible es que `b_splines` de efficient-kan **ya
está vectorizado** sobre `in_features`: devuelve (batch, n_fields, n_coef)
calculando todos los campos de una vez. Solo el producto posterior rompe la
independencia. Sustituyéndolo por `torch.bmm` con un tensor de pesos por
campo, se obtiene el mismo resultado con un solo kernel:

    bases:   (batch, n_fields, n_coef)
    weights: (n_fields, n_coef, embedding_dim)
    salida:  bmm -> (batch, n_fields, embedding_dim)

Equivalencia numérica
---------------------
`VectorizedKANEncoder.from_field_kans()` copia los pesos de un encoder ya
entrenado, de modo que la salida es idéntica (hasta precisión de máquina) a
la del encoder original. La optimización cambia la velocidad, nunca el
modelo: los tests lo verifican con tolerancia 1e-5 (diferencia medida:
2.4e-7, precisión de float32).

Cuándo ayuda y cuándo no (medido, no supuesto)
----------------------------------------------
En **CPU** la versión vectorizada es ~1.3x MÁS LENTA: sin coste de
lanzamiento de kernel que amortizar, el bucle de 13 llamadas pequeñas gana
frente a calcular las bases B-spline de los 13 campos en tensores mayores.

En **GPU** se espera lo contrario, porque el coste fijo de 13 lanzamientos
secuenciales domina cuando el trabajo por kernel es diminuto. Esa es la
hipótesis que motiva este módulo y debe verificarse midiendo: usar
`kanrec.latency.compare_latency` sobre el modelo original y el vectorizado
en el hardware de destino, y quedarse con el que gane. No se asume que
vectorizar sea siempre mejor.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import KANNumericalEncoder


class VectorizedKANEncoder(nn.Module):
    """
    Equivalente vectorizado de KANNumericalEncoder.

    Mantiene los campos independientes (cada φⱼ ve solo xⱼ) pero los evalúa
    en un único kernel en lugar de un bucle de n_fields lanzamientos.

    Se construye a partir de un encoder entrenado con `from_field_kans`, no
    desde cero: su propósito es acelerar la inferencia de un modelo ya
    ajustado, no reemplazar el entrenamiento.
    """

    def __init__(self, num_fields: int, embedding_dim: int, grid_size: int,
                 spline_order: int, input_clip: float = 10.0):
        super().__init__()
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.input_clip = input_clip
        n_coef = grid_size + spline_order

        # grid: (n_fields, grid_size + 2*spline_order + 1)
        self.register_buffer(
            "grid", torch.zeros(num_fields, grid_size + 2 * spline_order + 1)
        )
        # base_weight: (n_fields, 1, embedding_dim) — la ruta SiLU
        self.base_weight = nn.Parameter(torch.zeros(num_fields, 1, embedding_dim))
        # spline_weight: (n_fields, n_coef, embedding_dim)
        self.spline_weight = nn.Parameter(torch.zeros(num_fields, n_coef, embedding_dim))
        self.base_activation = nn.SiLU()

    @classmethod
    def from_field_kans(cls, encoder: KANNumericalEncoder) -> "VectorizedKANEncoder":
        """Construye la versión vectorizada copiando los pesos de un encoder entrenado."""
        first = encoder.field_kans[0].layers[0]
        vec = cls(
            num_fields=encoder.num_fields,
            embedding_dim=encoder.embedding_dim,
            grid_size=encoder.grid_size,
            spline_order=encoder.spline_order,
            input_clip=getattr(encoder, "input_clip", 10.0),
        )
        with torch.no_grad():
            for j, kan in enumerate(encoder.field_kans):
                layer = kan.layers[0]
                vec.grid[j] = layer.grid[0].detach()
                # base_weight de KANLinear: (out_features, in_features=1)
                vec.base_weight[j] = layer.base_weight.detach().t()
                # scaled_spline_weight: (out_features, in_features=1, n_coef)
                scaled = layer.scaled_spline_weight.detach()   # aplica spline_scaler
                vec.spline_weight[j] = scaled[:, 0, :].t()
        return vec.to(first.base_weight.device)

    def _b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Bases B-spline para todos los campos a la vez.

        Réplica exacta de efficient_kan.KANLinear.b_splines, que ya opera
        vectorizado sobre la dimensión de campos.

        Args:
            x: (batch, n_fields)
        Returns:
            (batch, n_fields, grid_size + spline_order)
        """
        grid = self.grid                      # (n_fields, n_knots)
        x = x.unsqueeze(-1)                   # (batch, n_fields, 1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
        return bases.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_fields) valores numéricos normalizados
        Returns:
            (batch, n_fields, embedding_dim)
        """
        # Mismo winsorizado que el encoder original: sin él, los outliers de
        # Criteo (~700 sigmas) desbordan el embedding en float32.
        x = x.clamp(-self.input_clip, self.input_clip)

        # Ruta base: SiLU(x) * base_weight, por campo.
        base = self.base_activation(x).unsqueeze(-1)          # (B, F, 1)
        base_out = base * self.base_weight.squeeze(1)          # (B, F, D)

        # Ruta spline: bmm mantiene los campos independientes.
        # (F, B, n_coef) x (F, n_coef, D) -> (F, B, D)
        bases = self._b_splines(x).transpose(0, 1)             # (F, B, n_coef)
        spline_out = torch.bmm(bases, self.spline_weight)      # (F, B, D)

        return base_out + spline_out.transpose(0, 1)           # (B, F, D)


def vectorize_model(model, verbose: bool = True):
    """
    Devuelve una copia del modelo con el encoder numérico vectorizado.

    El modelo original no se modifica. La salida es numéricamente idéntica;
    solo cambia la velocidad de inferencia.

    Uso:
        fast = vectorize_model(model)
        # fast(x_num, x_cat) == model(x_num, x_cat), pero más rápido
    """
    import copy

    if not isinstance(model.numerical_encoder, KANNumericalEncoder):
        if verbose:
            print("El modelo no usa KANNumericalEncoder; se devuelve sin cambios.")
        return model

    fast = copy.deepcopy(model)
    fast.numerical_encoder = VectorizedKANEncoder.from_field_kans(model.numerical_encoder)
    fast.eval()
    if verbose:
        print(f"Encoder vectorizado: {model.numerical_encoder.num_fields} campos "
              f"en 1 kernel (antes {model.numerical_encoder.num_fields} secuenciales)")
    return fast
