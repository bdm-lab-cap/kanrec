"""
Encoder KAN vectorizado.

Motivación (medición, no intuición)
-----------------------------------
El perfilado de latencia sobre GPU T4 (memoria, Resultado 5) mostró que el
encoder numérico consumía la mayor parte del tiempo de inferencia del modelo
completo y que KAN-REC era varias veces más lento que la normalización
directa. Las cifras de referencia son las de la tabla de la memoria, medidas
con ``kanrec.latency.compare_latency`` sobre el checkpoint final; no se
duplican aquí para que no se desincronicen.

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
entrenado. La optimización cambia la velocidad, nunca el modelo. Dos niveles
de verificación, y conviene no confundirlos al reportar:

- Los tests unitarios comprueban equivalencia con tolerancia 1e-5 sobre
  modelos aleatorios en CPU (`tests/test_vectorized.py`).
- La equivalencia sobre el checkpoint real y en el hardware de servicio se
  mide con `torch.equal` / diferencia máxima absoluta en el notebook de
  cierre. Un `bmm` y un `F.linear` pueden redondear distinto en GPU, así
  que "bit a bit" sólo puede afirmarse si esa medición da exactamente 0;
  si da ~1e-7, la afirmación correcta es "idéntica hasta precisión de
  float32".

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


class VectorizedRawEncoder(nn.Module):
    """
    Equivalente vectorizado de ``RawNumericalEncoder`` (la baseline de
    normalización directa): las 13 proyecciones ``Linear(1, d)`` se evalúan
    en una única operación elemento a elemento.

    Existe por honestidad en la comparativa de latencia, no porque la
    baseline lo necesite: ``RawNumericalEncoder`` también recorre los
    campos en un bucle de Python, así que el sobrecoste "KAN vectorizado
    frente a raw" medido contra la baseline sin vectorizar compara un
    encoder optimizado con otro que no lo está. Con ambos vectorizados el
    sobrecoste que se reporta es el del método, no el del bucle.
    """

    def __init__(self, num_fields: int, embedding_dim: int):
        super().__init__()
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.zeros(num_fields, embedding_dim))
        self.bias = nn.Parameter(torch.zeros(num_fields, embedding_dim))

    @classmethod
    def from_raw(cls, encoder) -> "VectorizedRawEncoder":
        vec = cls(encoder.num_fields, encoder.proj[0].out_features)
        with torch.no_grad():
            for j, lin in enumerate(encoder.proj):
                vec.weight[j] = lin.weight.detach()[:, 0]   # Linear(1, d): weight (d, 1)
                vec.bias[j] = lin.bias.detach()
        return vec.to(encoder.proj[0].weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, F, 1) * (F, D) + (F, D) -> (B, F, D)
        return x.unsqueeze(-1) * self.weight + self.bias


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
    Devuelve una copia del modelo con el encoder numérico vectorizado
    (KAN o raw; AutoDis se devuelve sin cambios).

    El modelo original no se modifica. La salida es numéricamente idéntica;
    solo cambia la velocidad de inferencia.

    Uso:
        fast = vectorize_model(model)
        # fast(x_num, x_cat) == model(x_num, x_cat), pero más rápido
    """
    import copy

    from .baselines import RawNumericalEncoder

    enc = model.numerical_encoder
    if isinstance(enc, KANNumericalEncoder):
        new_enc = VectorizedKANEncoder.from_field_kans(enc)
    elif isinstance(enc, RawNumericalEncoder):
        new_enc = VectorizedRawEncoder.from_raw(enc)
    else:
        if verbose:
            print(f"{type(enc).__name__} no tiene versión vectorizada; se devuelve sin cambios.")
        return model

    fast = copy.deepcopy(model)
    fast.numerical_encoder = new_enc
    fast.eval()
    if verbose:
        print(f"Encoder vectorizado ({type(enc).__name__}): {enc.num_fields} campos "
              f"en 1 kernel (antes {enc.num_fields} lanzamientos secuenciales)")
    return fast
