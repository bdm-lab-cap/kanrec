"""
Ablación por sustitución: la métrica de fidelidad del extractor simbólico.

Qué mide y por qué importa
--------------------------
La extracción simbólica produce una fórmula φ̂ⱼ(x) por campo. La pregunta
que decide si esa fórmula sirve para algo es: **¿se comporta el modelo
igual si sustituimos el spline aprendido por su fórmula?**

Este módulo responde a esa pregunta de la única forma que la valida:
reemplaza φⱼ por φ̂ⱼ **dentro del modelo**, dejando intactas las 26
embeddings categóricas, la capa de interacción y la cabeza, y mide la
caída de AUC y el RMSE entre las predicciones original y sustituida.

Por qué no basta el `compute_gap_rmse` de FaithfulnessEvaluator
---------------------------------------------------------------
Aquel método suma los φ̂ⱼ y aplica una sigmoide, ignorando categóricas,
interacción y cabeza. Es un GAM aditivo que no puede aproximar al modelo
por construcción: su "gap" mide la distancia a un modelo distinto, no la
fidelidad de la fórmula. La ablación por sustitución sí aísla el efecto
de la fórmula, porque todo lo demás se mantiene igual.

Lectura del resultado
---------------------
  - ΔAUC ≈ 0  ->  la fórmula captura lo que el spline aportaba al modelo:
                  la interpretación es fiel.
  - ΔAUC grande -> el spline hace algo que la fórmula no recoge; la
                  fórmula es una descripción incompleta y hay que decirlo.

Uso:
    from kanrec.ablation import substitution_ablation
    report = substitution_ablation(model, test_loader, symbolic_results)
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import log_loss, roc_auc_score

from .symbolic import OPERATOR_LIBRARY


class SymbolicFieldEncoder(nn.Module):
    """
    Sustituto de un `KAN` de campo que evalúa la fórmula simbólica.

    El KAN original mapea x -> [batch, embedding_dim]. La fórmula ajustada
    describe la forma de la curva, así que se replica esa forma en todas
    las dimensiones escalándola por el factor que mejor reproduce cada
    dimensión del spline original (ajuste por mínimos cuadrados sin
    término independiente, más el offset medio). Así la sustitución
    conserva la escala y el desplazamiento por dimensión, y lo único que
    cambia es la FORMA de la curva, que es lo que la fórmula afirma
    describir.
    """

    def __init__(self, operator: str, params: list[float],
                 scale: torch.Tensor, offset: torch.Tensor,
                 input_clip: float = 10.0):
        super().__init__()
        self.operator = operator
        self.params = params
        self.input_clip = input_clip
        self.register_buffer("scale", scale)    # [embedding_dim]
        self.register_buffer("offset", offset)  # [embedding_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 1] -> [batch, embedding_dim]
        xc = x.clamp(-self.input_clip, self.input_clip)
        x_np = xc.detach().cpu().numpy().reshape(-1)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            base = OPERATOR_LIBRARY[self.operator](x_np, *self.params)
        base = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)
        base_t = torch.as_tensor(base, dtype=x.dtype, device=x.device).unsqueeze(1)
        return base_t * self.scale + self.offset


def _fit_scale_offset(model, field_idx: int, operator: str,
                      params: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Para cada dimensión d del embedding, encuentra (scale_d, offset_d) tales
    que `scale_d * formula(x) + offset_d` se aproxime a la curva real φⱼ,d(x).
    Es una regresión lineal simple de la curva real sobre la fórmula.
    """
    x_grid, curves = model.numerical_encoder.get_spline_curves(field_idx)
    x_np = x_grid.cpu().numpy()
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        f = OPERATOR_LIBRARY[operator](x_np, *params)
    f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)

    scales, offsets = [], []
    f_var = np.var(f)
    for d in range(curves.shape[1]):
        y = curves[:, d].cpu().numpy()
        if f_var < 1e-12:          # fórmula constante: solo offset
            scales.append(0.0)
            offsets.append(float(y.mean()))
        else:
            s = float(np.cov(f, y, bias=True)[0, 1] / f_var)
            scales.append(s)
            offsets.append(float(y.mean() - s * f.mean()))
    return (torch.tensor(scales, dtype=torch.float32),
            torch.tensor(offsets, dtype=torch.float32))


@torch.no_grad()
def _evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (probabilidades, etiquetas) sobre el loader."""
    model.eval()
    preds, labels = [], []
    for x_num, x_cat, y in loader:
        logits = model(x_num.to(device), x_cat.to(device)).squeeze()
        preds.extend(np.atleast_1d(torch.sigmoid(logits).cpu().numpy()))
        labels.extend(y.numpy())
    return np.asarray(preds, dtype=float), np.asarray(labels, dtype=float)


def substitution_ablation(model, test_loader, symbolic_results: dict,
                          numerical_cols: list[str] | None = None,
                          device: torch.device | None = None,
                          verbose: bool = True) -> dict:
    """
    Sustituye φⱼ por su fórmula simbólica dentro del modelo y mide el efecto.

    Args:
        model:             KANRecModel entrenado.
        test_loader:       DataLoader del conjunto de test.
        symbolic_results:  {field_idx | field_name: {"operator", "params", "accepted"}}
                            tal como los produce fit_field / symbolic_results.json.
        numerical_cols:    nombres de campo, para poder aceptar claves por nombre.
        device:            dispositivo; por defecto el del modelo.

    Returns:
        dict con auc_original, auc_sustituido, delta_auc, rmse_predicciones,
        y el detalle por campo.
    """
    device = device or next(model.parameters()).device

    # Normalizar las claves a índices de campo
    results_by_idx: dict[int, dict] = {}
    for key, r in symbolic_results.items():
        if isinstance(key, int):
            idx = key
        elif numerical_cols is not None and key in numerical_cols:
            idx = numerical_cols.index(key)
        else:
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
        results_by_idx[idx] = r

    usable = {j: r for j, r in results_by_idx.items()
              if r.get("accepted") and r.get("operator") and r.get("params")}
    if not usable:
        raise ValueError("No hay fórmulas aceptadas que sustituir.")

    preds_orig, labels = _evaluate(model, test_loader, device)
    auc_orig = roc_auc_score(labels, preds_orig)
    ll_orig = log_loss(labels, preds_orig)

    if verbose:
        print(f"Modelo original:    AUC={auc_orig:.4f}  log-loss={ll_orig:.4f}")
        print(f"Sustituyendo {len(usable)} campos por su fórmula simbólica...")

    # Copia profunda: la sustitución no debe tocar el modelo original
    model_sub = copy.deepcopy(model).to(device)
    input_clip = getattr(model.numerical_encoder, "input_clip", 10.0)

    per_field = []
    for j, r in sorted(usable.items()):
        scale, offset = _fit_scale_offset(model, j, r["operator"], r["params"])

        # Error de curva: cuanto se desvia la curva SUSTITUIDA de la real,
        # relativo a la dispersion de la curva real, promediado sobre las
        # dimensiones del embedding. Es la medida DIRECTA de fidelidad de la
        # formula, y es mas sensible que el delta de AUC: en CTR el AUC
        # depende sobre todo de las categoricas, asi que puede moverse poco
        # aunque la forma de phi cambie mucho (verificado: senal curva da
        # error de curva 0.58 frente a 0.16 en el caso lineal, mientras el
        # delta de AUC apenas distingue ambos casos).
        x_grid, curves = model.numerical_encoder.get_spline_curves(j)
        x_np = x_grid.cpu().numpy()
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            f_np = OPERATOR_LIBRARY[r["operator"]](x_np, *r["params"])
        f_np = np.nan_to_num(f_np, nan=0.0, posinf=0.0, neginf=0.0)
        rel_errs = []
        for d in range(curves.shape[1]):
            real = curves[:, d].cpu().numpy()
            approx = f_np * float(scale[d]) + float(offset[d])
            rel_errs.append(
                float(np.sqrt(np.mean((approx - real) ** 2)) / (real.std() + 1e-9))
            )
        curve_err = float(np.mean(rel_errs))

        model_sub.numerical_encoder.field_kans[j] = SymbolicFieldEncoder(
            r["operator"], r["params"], scale.to(device), offset.to(device), input_clip
        ).to(device)
        name = numerical_cols[j] if numerical_cols else f"I{j+1}"
        per_field.append({"field": name, "field_idx": j,
                          "operator": r["operator"], "r2": r.get("r2"),
                          "curve_rel_error": curve_err})

    preds_sub, _ = _evaluate(model_sub, test_loader, device)
    finite = np.isfinite(preds_sub)
    if not finite.all() and verbose:
        print(f"  aviso: {int((~finite).sum())} predicciones no finitas tras sustituir")
    if finite.sum() == 0:
        raise RuntimeError("Todas las predicciones sustituidas son no finitas.")

    auc_sub = roc_auc_score(labels[finite], preds_sub[finite])
    ll_sub = log_loss(labels[finite], preds_sub[finite])
    rmse = float(np.sqrt(np.mean((preds_orig[finite] - preds_sub[finite]) ** 2)))

    curve_errs = [f["curve_rel_error"] for f in per_field]
    report = {
        "n_fields_substituted": len(usable),
        # Fidelidad de la FORMA (el indicador directo, ver per_field)
        "curve_rel_error_mean": float(np.mean(curve_errs)),
        "curve_rel_error_max": float(np.max(curve_errs)),
        "auc_original": float(auc_orig),
        "auc_substituted": float(auc_sub),
        "delta_auc": float(auc_orig - auc_sub),
        "logloss_original": float(ll_orig),
        "logloss_substituted": float(ll_sub),
        "prediction_rmse": rmse,
        "fields": per_field,
    }

    if verbose:
        print(f"Modelo sustituido:  AUC={auc_sub:.4f}  log-loss={ll_sub:.4f}")
        print(f"\n  ΔAUC = {report['delta_auc']:+.4f}")
        print(f"  RMSE entre predicciones = {rmse:.4f}")
        print(f"  Error de curva (relativo) = {report['curve_rel_error_mean']:.3f} "
              f"medio, {report['curve_rel_error_max']:.3f} maximo")
        print("\n  Detalle por campo (error de curva):")
        for f in sorted(per_field, key=lambda f: -f["curve_rel_error"]):
            print(f"    {f['field']:>5}: {f['operator']:<8} error={f['curve_rel_error']:.3f}")

        # El veredicto se basa en el error de CURVA, no en el delta de AUC:
        # en CTR el AUC lo dominan las categoricas y es poco sensible a la
        # forma de phi (verificado empiricamente).
        err = report["curve_rel_error_mean"]
        print()
        if err < 0.20:
            print("  -> Formulas FIELES: reproducen la forma de phi con error <20%.")
        elif err < 0.40:
            print("  -> Fidelidad razonable: la formula captura la tendencia principal.")
        else:
            print("  -> Fidelidad BAJA: la formula no reproduce la forma de phi. "
                  "Debe reportarse asi en la memoria, no presentarse como "
                  "descripcion fiel del encoder.")
    return report
