"""
Mide la latencia de los cinco modelos comparados y la equivalencia numérica
entre cada encoder y su versión vectorizada.

Se miden raw, raw vectorizado, AutoDis, KAN y KAN vectorizado. El sobrecoste
relevante es KAN-vec / raw-vec: con ambas baselines vectorizadas, la
diferencia medida es la del método y no la del bucle por campo.

La equivalencia se comprueba con `torch.equal` sobre el checkpoint real y en
el hardware de servicio, no solo con la tolerancia de los tests unitarios.

Uso (GPU T4, con los checkpoints de la comparativa disponibles):

    python experiments/latency_report.py \
        --ckpt-dir checkpoints --seed 42 --grid-size 10 \
        --n-cat 26 --cardinalities cardinalities.json --out resultados/latency.json

`cardinalities.json` es opcional: sin él se infieren del propio checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from kanrec.latency import compare_latency
from kanrec.serving import infer_architecture, load_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid-size", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--n-runs", type=int, default=100)
    ap.add_argument("--out", default="resultados/latency.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'})")

    paths = {
        "raw": f"{args.ckpt_dir}/best_raw_gs{args.grid_size}_s{args.seed}.pt",
        "autodis": f"{args.ckpt_dir}/best_autodis_gs{args.grid_size}_s{args.seed}.pt",
        "kan-bspline": f"{args.ckpt_dir}/best_kan-bspline_gs{args.grid_size}_s{args.seed}.pt",
    }
    for name, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"falta el checkpoint de {name}: {p}")

    models = {
        "raw": load_model(paths["raw"], device, vectorize=False),
        "raw-vec": load_model(paths["raw"], device, vectorize=True),
        "autodis": load_model(paths["autodis"], device, vectorize=False),
        "kan-bspline": load_model(paths["kan-bspline"], device, vectorize=False),
        "kan-vec": load_model(paths["kan-bspline"], device, vectorize=True),
    }
    arch = infer_architecture(torch.load(paths["kan-bspline"], map_location="cpu"))

    # Lote representativo: ruido estandarizado con colas (winsorizado por el encoder)
    torch.manual_seed(0)
    x_num = (torch.randn(args.batch, arch["num_numerical"]) * 3).to(device)
    x_cat = torch.stack(
        [torch.randint(0, c + 1, (args.batch,)) for c in arch["cat_cardinalities"]], dim=1
    ).to(device)

    # ── Equivalencia numérica original / vectorizado ────────────────────────
    equivalence = {}
    for base, vec in (("raw", "raw-vec"), ("kan-bspline", "kan-vec")):
        with torch.no_grad():
            a, b = models[base](x_num, x_cat), models[vec](x_num, x_cat)
        equivalence[vec] = {
            "torch_equal": bool(torch.equal(a, b)),
            "max_abs_diff": float((a - b).abs().max()),
        }
        print(f"{vec:>8}: torch.equal={equivalence[vec]['torch_equal']}  "
              f"max|Δ|={equivalence[vec]['max_abs_diff']:.2e}")

    # ── Latencias ───────────────────────────────────────────────────────────
    results = compare_latency(models, x_num, x_cat, n_runs=args.n_runs, verbose=True)

    def ratio(a: str, b: str) -> float:
        return results[a]["full_ms_mean"] / results[b]["full_ms_mean"]

    summary = {
        "device": device, "batch": args.batch, "n_runs": args.n_runs,
        "speedup_kan_vectorization": ratio("kan-bspline", "kan-vec"),
        "speedup_kan_encoder_only": results["kan-bspline"]["encoder_ms_mean"] / results["kan-vec"]["encoder_ms_mean"],
        "overhead_kan_vs_raw_vectorized": ratio("kan-vec", "raw-vec"),
        "overhead_kan_vs_raw_unvectorized": ratio("kan-vec", "raw"),
        "kan_vec_vs_autodis": ratio("autodis", "kan-vec"),
        "encoder_share_kan_before": results["kan-bspline"]["encoder_share"],
        "encoder_share_kan_after": results["kan-vec"]["encoder_share"],
    }
    print("\nResumen:")
    for k, v in summary.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"latency": results, "equivalence": equivalence, "summary": summary}, f, indent=2)
    print(f"\nguardado en {args.out}")


if __name__ == "__main__":
    main()
