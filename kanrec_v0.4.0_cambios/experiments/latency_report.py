"""
Regenera la tabla de latencias de la memoria (Resultado 5) a partir de los
checkpoints reales y mide la equivalencia original/vectorizado.

Por qué existe
--------------
1. El sobrecoste "KAN vectorizado frente a normalización directa" se había
   medido contra una baseline raw SIN vectorizar (un bucle de 13
   `Linear(1, d)`). Aquí se miden los cinco modelos: raw, raw vectorizado,
   AutoDis, KAN, KAN vectorizado. El sobrecoste que hay que reportar es
   KAN-vec / raw-vec: dos encoders optimizados, no uno contra un bucle.

2. La memoria afirma que la salida vectorizada es "idéntica bit a bit". Eso
   sólo puede decirse si `torch.equal` devuelve True sobre el checkpoint
   real y en el hardware de servicio. Este script lo mide y lo escribe en
   el JSON de salida; la frase de la memoria debe copiar ese valor.

Uso (Colab, GPU T4, tras 04 / con los checkpoints descargados):

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

    # ── Equivalencia: lo que de verdad se puede afirmar ─────────────────────
    equivalence = {}
    for base, vec in (("raw", "raw-vec"), ("kan-bspline", "kan-vec")):
        with torch.no_grad():
            a, b = models[base](x_num, x_cat), models[vec](x_num, x_cat)
        equivalence[vec] = {
            "torch_equal": bool(torch.equal(a, b)),
            "max_abs_diff": float((a - b).abs().max()),
            "claim": ("idéntica bit a bit" if torch.equal(a, b)
                      else f"idéntica hasta precisión de float32 (máx {float((a - b).abs().max()):.1e})"),
        }
        print(f"{vec:>8}: torch.equal={equivalence[vec]['torch_equal']}  "
              f"max|Δ|={equivalence[vec]['max_abs_diff']:.2e}  → «{equivalence[vec]['claim']}»")

    # ── Latencias ───────────────────────────────────────────────────────────
    results = compare_latency(models, x_num, x_cat, n_runs=args.n_runs, verbose=True)

    def ratio(a: str, b: str) -> float:
        return results[a]["full_ms_mean"] / results[b]["full_ms_mean"]

    summary = {
        "device": device, "batch": args.batch, "n_runs": args.n_runs,
        "speedup_kan_vectorization": ratio("kan-bspline", "kan-vec"),
        "speedup_kan_encoder_only": results["kan-bspline"]["encoder_ms_mean"] / results["kan-vec"]["encoder_ms_mean"],
        "overhead_kan_vs_raw_UNFAIR (raw sin vectorizar)": ratio("kan-vec", "raw"),
        "overhead_kan_vs_raw (ambos vectorizados) <- REPORTAR ESTE": ratio("kan-vec", "raw-vec"),
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
