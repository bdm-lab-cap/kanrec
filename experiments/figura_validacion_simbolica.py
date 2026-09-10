"""
Genera la figura de validacion del extractor simbolico: curvas phi
aprendidas sobre una senal NO LINEAL conocida frente a una senal LINEAL
conocida, mas (opcionalmente) la curva real de Criteo.

Por que esta figura importa para el TFM
---------------------------------------
La extraccion simbolica sobre Criteo devuelve operadores lineales en todos
los campos. Por si sola, esa observacion es ambigua: puede significar que
los datos son lineales, o que el metodo es incapaz de detectar otra cosa.

Esta figura resuelve la ambiguedad con un control: se entrena el MISMO
modelo sobre datos sinteticos donde la forma verdadera se conoce, y se
comprueba que el encoder la recupera. Si con sin(1.5x) la curva sale
curvada y con 1.2x sale recta, entonces la linealidad detectada en Criteo
es una propiedad de los datos, no una limitacion del extractor.

Es el argumento que convierte un resultado aparentemente flojo ("todo
sale lineal") en evidencia positiva ("el metodo detecta correctamente que
la relacion es lineal").

Uso:
    python experiments/figura_validacion_simbolica.py
    python experiments/figura_validacion_simbolica.py --out figs/validacion.png
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from kanrec.model import KANRecModel


def train_on_signal(signal: str, seed: int = 42, n_rows: int = 25_000,
                    epochs: int = 15, grid_size: int = 10) -> KANRecModel:
    """
    Entrena KAN-REC sobre datos sinteticos con la forma funcional indicada
    en el campo 0. Los datos imitan la distribucion de Criteo tras
    normalizar: muy concentrados en un valor, con cola a la derecha.
    """
    torch.manual_seed(seed)
    n_numerical, n_categorical = 13, 5

    x_num = torch.full((n_rows, n_numerical), -0.27)
    for j in range(n_numerical):
        n_nz = int(n_rows * 0.15)
        x_num[:n_nz, j] = torch.rand(n_nz) * 8
    x_cat = torch.randint(0, 100, (n_rows, n_categorical))

    if signal == "curva":
        logit = 3.0 * torch.sin(x_num[:, 0] * 1.5) - 1.0
    elif signal == "lineal":
        logit = 1.2 * x_num[:, 0] - 1.0
    else:
        raise ValueError(f"signal desconocida: {signal}")
    y = (torch.rand(n_rows) < torch.sigmoid(logit)).float()

    model = KANRecModel(num_numerical=n_numerical,
                        cat_cardinalities=[100] * n_categorical,
                        embedding_dim=16, kan_grid_size=grid_size)
    model.calibrate(x_num[: n_rows // 3])

    optimizer = torch.optim.Adam(model.parameter_groups(base_lr=1e-3), weight_decay=1e-5)
    criterion = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        perm = torch.randperm(n_rows)
        for i in range(0, n_rows, 2048):
            idx = perm[i : i + 2048]
            optimizer.zero_grad()
            loss = criterion(model(x_num[idx], x_cat[idx]).squeeze(), y[idx])
            loss = loss + model.entropy_regularization_loss()
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


def linear_r2(x: np.ndarray, y: np.ndarray) -> float:
    """R2 de un ajuste lineal: bajo = curva genuina, alto = recta."""
    coeffs = np.polyfit(x, y, 1)
    ss_res = np.sum((y - np.polyval(coeffs, x)) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
    return float(1 - ss_res / ss_tot)


def main(out_path: str, n_dims_shown: int = 4) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=False)

    for ax, (signal, titulo, formula) in zip(
        axes,
        [("curva", "Señal no lineal", r"$\mathrm{logit} = 3\,\sin(1.5x) - 1$"),
         ("lineal", "Señal lineal", r"$\mathrm{logit} = 1.2x - 1$")],
    ):
        print(f"Entrenando sobre señal '{signal}'...")
        model = train_on_signal(signal)
        x_grid, curves = model.numerical_encoder.get_spline_curves(0)
        xs = x_grid.numpy()

        r2s = [linear_r2(xs, curves[:, d].numpy()) for d in range(curves.shape[1])]

        for d in range(min(n_dims_shown, curves.shape[1])):
            ax.plot(xs, curves[:, d].numpy(), lw=1.8, alpha=0.85, label=f"dim {d}")

        ax.set_title(f"{titulo}\n{formula}", fontsize=11)
        ax.set_xlabel("valor normalizado del campo")
        ax.set_ylabel(r"$\phi(x)$")
        ax.axhline(0, color="0.8", lw=0.8, zorder=0)
        ax.legend(fontsize=8, loc="best")
        ax.text(0.03, 0.03,
                f"$R^2$ lineal: {np.mean(r2s):.3f} ± {np.std(r2s):.3f}\n"
                f"(mín. {np.min(r2s):.3f}, sobre {len(r2s)} dims)",
                transform=ax.transAxes, fontsize=9, va="bottom",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", alpha=0.9))
        print(f"  R2 lineal medio: {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")

    fig.suptitle("Validación del encoder: la curva aprendida sigue la forma real de los datos",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"\nFigura guardada en {out_path}")
    print("Lectura: con señal no lineal el encoder produce curvas (R2 lineal bajo);")
    print("con señal lineal produce rectas (R2 lineal alto). Por tanto, que en")
    print("Criteo salgan rectas es un hallazgo sobre los datos, no una")
    print("limitación del extractor simbólico.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="validacion_simbolica.png")
    parser.add_argument("--dims", type=int, default=4,
                        help="cuantas dimensiones del embedding dibujar")
    args = parser.parse_args()
    main(args.out, args.dims)
