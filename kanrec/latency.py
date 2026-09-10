"""
Medición de latencia de inferencia por encoder.

La propuesta de TFM (tabla 7.2) compromete "Latencia (ms/batch) — Eficiencia
del encoder — Batch size 4096 en GPU T4; media sobre 100 batches" como una de
las métricas de evaluación. Este módulo la implementa.

Por qué importa: el argumento a favor de KAN-REC no es el AUC (equipara a los
baselines), sino que ofrece interpretabilidad a un coste asumible. Sin medir
ese coste, la afirmación queda sin respaldo. Y la propuesta afirmaba además
que EfficientKAN sería "igual o inferior" en latencia a AutoDis, al no
requerir la suma ponderada de meta-embeddings: eso es contrastable.

Metodología: se separa el tiempo del ENCODER NUMÉRICO del tiempo del modelo
completo, porque es el encoder lo que se compara. Se descartan iteraciones de
calentamiento (la primera pasada en GPU incluye compilación de kernels) y se
sincroniza CUDA antes de cada medición, sin lo cual se mide el tiempo de
encolado y no el de ejecución.
"""
from __future__ import annotations

import time

import numpy as np
import torch


@torch.no_grad()
def measure_latency(model, x_num: torch.Tensor, x_cat: torch.Tensor,
                    n_warmup: int = 10, n_runs: int = 100,
                    device: torch.device | None = None) -> dict:
    """
    Mide la latencia de inferencia de un modelo y de su encoder numérico.

    Args:
        model:    modelo a medir (CTRModel o KANRecModel).
        x_num:    [batch, num_numerical] un batch representativo.
        x_cat:    [batch, num_categorical].
        n_warmup: iteraciones de calentamiento descartadas.
        n_runs:   iteraciones medidas.

    Returns:
        dict con ms/batch (media, desviación, p50, p95) para el modelo
        completo y para el encoder numérico por separado, más el throughput.
    """
    device = device or next(model.parameters()).device
    model.eval()
    x_num, x_cat = x_num.to(device), x_cat.to(device)
    is_cuda = device.type == "cuda"

    def _sync():
        if is_cuda:
            torch.cuda.synchronize()

    def _time(fn) -> list[float]:
        for _ in range(n_warmup):     # calentamiento: kernels, cachés
            fn()
        _sync()
        times = []
        for _ in range(n_runs):
            _sync()
            t0 = time.perf_counter()
            fn()
            _sync()                    # sin esto se mide el encolado, no la ejecución
            times.append((time.perf_counter() - t0) * 1000.0)
        return times

    full = _time(lambda: model(x_num, x_cat))
    enc = _time(lambda: model.numerical_encoder(x_num))

    batch_size = x_num.size(0)
    return {
        "batch_size": batch_size,
        "device": str(device),
        "n_runs": n_runs,
        "full_ms_mean": float(np.mean(full)),
        "full_ms_std": float(np.std(full)),
        "full_ms_p50": float(np.percentile(full, 50)),
        "full_ms_p95": float(np.percentile(full, 95)),
        "encoder_ms_mean": float(np.mean(enc)),
        "encoder_ms_std": float(np.std(enc)),
        "encoder_ms_p95": float(np.percentile(enc, 95)),
        "encoder_share": float(np.mean(enc) / np.mean(full)) if np.mean(full) > 0 else 0.0,
        "throughput_rows_per_s": float(batch_size / (np.mean(full) / 1000.0)),
    }


def compare_latency(models: dict, x_num: torch.Tensor, x_cat: torch.Tensor,
                    n_warmup: int = 10, n_runs: int = 100,
                    verbose: bool = True) -> dict:
    """
    Mide la latencia de varios encoders y los compara con el más rápido.

    Args:
        models: {nombre: modelo}, p.ej. {"raw": m1, "autodis": m2, "kan-bspline": m3}
    """
    results = {name: measure_latency(m, x_num, x_cat, n_warmup, n_runs)
               for name, m in models.items()}

    if verbose:
        fastest = min(results.values(), key=lambda r: r["full_ms_mean"])["full_ms_mean"]
        print(f"\nLatencia de inferencia (batch={x_num.size(0)}, "
              f"{next(iter(results.values()))['device']}, media de {n_runs} runs)")
        print(f"{'encoder':>12} {'modelo ms':>12} {'encoder ms':>12} "
              f"{'% encoder':>10} {'vs mejor':>9} {'filas/s':>12}")
        print("-" * 72)
        for name, r in sorted(results.items(), key=lambda kv: kv[1]["full_ms_mean"]):
            print(f"{name:>12} {r['full_ms_mean']:>9.2f}±{r['full_ms_std']:<4.2f} "
                  f"{r['encoder_ms_mean']:>9.2f}±{r['encoder_ms_std']:<4.2f} "
                  f"{r['encoder_share']:>9.1%} "
                  f"{r['full_ms_mean']/fastest:>8.2f}x "
                  f"{r['throughput_rows_per_s']:>12,.0f}")
    return results
