"""
Servicio del modelo: cargar un checkpoint y puntuar impresiones, en local o
desde Spark (procesamiento del stream, notebook 06).

Carga autocontenida
-------------------
``infer_architecture`` deduce el encoder y los hiperparámetros estructurales
de las formas del ``state_dict``, de modo que un checkpoint puede cargarse
sin reconstruir el contexto de entrenamiento (número de campos, dimensión de
embedding, ``grid_size``, cardinalidades). El grid calibrado viaja dentro del
checkpoint como buffer. Para el encoder KAN se devuelve por defecto la
versión vectorizada (``kanrec.vectorized``), que es la que se sirve.

Manifiesto
----------
``write_manifest`` escribe junto al checkpoint un JSON con la versión del
paquete, el commit, las columnas, las cardinalidades, los hiperparámetros y
el hash SHA-256 del propio checkpoint y de ``scaler_stats.json``.
``check_manifest`` los verifica antes de servir: es lo que hace comprobable
—y no solo convencional— que el stream puntúa con los mismos estadísticos de
normalización con los que se entrenó.

Índices fuera de rango
----------------------
04 calcula la cardinalidad como ``max(train, val, test) + 1`` y 06 asigna a
las categorías no vistas ``len(categorías de train)``; ambos caben en el
embedding. Un índice mayor abortaría la inferencia con un *device-side
assert*, así que en servicio se recorta y se cuenta
(``Scorer.n_clamped``) en lugar de abortar.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone

import numpy as np

# ── Arquitectura desde el state_dict ─────────────────────────────────────────

def infer_architecture(state_dict: dict) -> dict:
    """
    Deduce encoder e hiperparámetros estructurales de las formas del
    ``state_dict`` de un ``CTRModel`` / ``KANRecModel``.

    Returns:
        dict con: encoder ('kan-bspline' | 'raw' | 'autodis'), num_numerical,
        embedding_dim, cat_cardinalities y, para KAN, grid_size y
        spline_order; para AutoDis, num_buckets.
    """
    keys = list(state_dict.keys())
    cat_keys = sorted(
        (k for k in keys if k.startswith("cat_embeddings.") and k.endswith(".weight")),
        key=lambda k: int(k.split(".")[1]),
    )
    if not cat_keys:
        raise ValueError("state_dict sin cat_embeddings: no es un CTRModel de kanrec")
    # nn.Embedding(card + 1, D): se resta 1 para volver a pasarlo al constructor.
    cat_cardinalities = [int(state_dict[k].shape[0]) - 1 for k in cat_keys]
    embedding_dim = int(state_dict[cat_keys[0]].shape[1])

    kan_grids = [k for k in keys if k.startswith("numerical_encoder.field_kans.")
                 and k.endswith(".layers.0.grid")]
    if kan_grids:
        num_numerical = len(kan_grids)
        grid_len = int(state_dict[kan_grids[0]].shape[1])
        sw = state_dict["numerical_encoder.field_kans.0.layers.0.spline_weight"]
        n_coef = int(sw.shape[-1])
        # grid_len = grid_size + 2*order + 1 ; n_coef = grid_size + order
        spline_order = grid_len - n_coef - 1
        grid_size = n_coef - spline_order
        return {"encoder": "kan-bspline", "num_numerical": num_numerical,
                "embedding_dim": embedding_dim, "cat_cardinalities": cat_cardinalities,
                "grid_size": grid_size, "spline_order": spline_order}

    raw_keys = [k for k in keys if k.startswith("numerical_encoder.proj.") and k.endswith(".weight")]
    if raw_keys:
        return {"encoder": "raw", "num_numerical": len(raw_keys),
                "embedding_dim": embedding_dim, "cat_cardinalities": cat_cardinalities}

    ad_keys = [k for k in keys if k.startswith("numerical_encoder.project.") and k.endswith(".weight")]
    if ad_keys:
        return {"encoder": "autodis", "num_numerical": len(ad_keys),
                "embedding_dim": embedding_dim, "cat_cardinalities": cat_cardinalities,
                "num_buckets": int(state_dict[ad_keys[0]].shape[0])}

    raise ValueError("No se reconoce el encoder numérico del state_dict")


def load_model(ckpt_path: str, device: str = "cpu", vectorize: bool = True):
    """
    Carga un checkpoint de kanrec sin necesitar ningún dato de entrenamiento.

    Args:
        ckpt_path: fichero ``.pt`` con el ``state_dict``.
        device:    'cpu' o 'cuda'.
        vectorize: si el encoder es KAN (o raw), sustituirlo por su versión
                   vectorizada. La salida es la misma; sólo cambia la
                   velocidad (ver kanrec.vectorized).
    """
    import torch
    from .baselines import build_model
    from .vectorized import vectorize_model

    state = torch.load(ckpt_path, map_location="cpu")
    arch = infer_architecture(state)
    kwargs = {k: v for k, v in arch.items() if k in ("num_numerical", "cat_cardinalities", "embedding_dim")}
    if arch["encoder"] == "kan-bspline":
        kwargs.update(kan_grid_size=arch["grid_size"], kan_spline_order=arch["spline_order"])
    elif arch["encoder"] == "autodis":
        kwargs.update(autodis_num_buckets=arch["num_buckets"])
    model = build_model(arch["encoder"], **kwargs)
    model.load_state_dict(state)
    model.eval()
    if vectorize:
        model = vectorize_model(model, verbose=False)
    return model.to(device)


# ── Scoring ──────────────────────────────────────────────────────────────────

class Scorer:
    """
    Puntúa lotes de impresiones ya normalizadas (misma transformación que
    train: log1p, estandarización con los estadísticos de train e índices
    categóricos con los mapas de train).

    Args:
        model:            modelo cargado con ``load_model``.
        numerical_cols:   nombres de las columnas numéricas, en el orden del modelo.
        categorical_cols: nombres de las columnas de índice categórico, en orden.
    """

    def __init__(self, model, numerical_cols: list[str], categorical_cols: list[str]):
        import torch
        self.model = model.eval()
        self.numerical_cols = list(numerical_cols)
        self.categorical_cols = list(categorical_cols)
        self.device = next(model.parameters()).device
        self._max_idx = torch.tensor(
            [emb.num_embeddings - 1 for emb in model.cat_embeddings],
            dtype=torch.long, device=self.device,
        )
        if len(self.categorical_cols) != len(self._max_idx):
            raise ValueError(
                f"el modelo tiene {len(self._max_idx)} embeddings categóricos y se "
                f"pasaron {len(self.categorical_cols)} columnas"
            )
        self.n_clamped = 0

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, numerical_cols, categorical_cols,
                        device: str = "cpu") -> "Scorer":
        return cls(load_model(ckpt_path, device=device), numerical_cols, categorical_cols)

    def predict_proba(self, num: np.ndarray, cat: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        """P(click) para arrays [n, n_num] float y [n, n_cat] int."""
        import torch
        num = np.asarray(num, dtype=np.float32)
        cat = np.asarray(cat, dtype=np.int64)
        if num.ndim != 2 or cat.ndim != 2 or num.shape[0] != cat.shape[0]:
            raise ValueError("num y cat deben ser 2D con el mismo número de filas")
        out = np.empty(num.shape[0], dtype=np.float32)
        with torch.no_grad():
            for s in range(0, num.shape[0], batch_size):
                xn = torch.from_numpy(np.nan_to_num(num[s:s + batch_size])).to(self.device)
                xc = torch.from_numpy(cat[s:s + batch_size]).to(self.device)
                clamped = xc.clamp(min=0).minimum(self._max_idx)
                self.n_clamped += int((clamped != xc).any(dim=1).sum())
                out[s:s + batch_size] = self.model.predict_proba(xn, clamped).squeeze(-1).cpu().numpy()
        return out

    def predict_frame(self, pdf, batch_size: int = 8192) -> np.ndarray:
        """P(click) para un pandas.DataFrame con las columnas del modelo."""
        return self.predict_proba(
            pdf[self.numerical_cols].to_numpy(dtype=np.float32, na_value=0.0),
            pdf[self.categorical_cols].fillna(0).to_numpy(dtype=np.int64),
            batch_size=batch_size,
        )


_SCORER_CACHE: dict[tuple, Scorer] = {}


def score_spark(df, ckpt_path: str, numerical_cols: list[str], categorical_cols: list[str],
                output_col: str = "p_click", batch_size: int = 8192):
    """
    Añade ``output_col`` = P(click) a un DataFrame de Spark usando
    ``mapInPandas``. El modelo se carga una vez por ejecutor (caché de
    módulo) y se reutiliza entre particiones.

    ``ckpt_path`` debe ser accesible desde los ejecutores; en Fabric el
    lakehouse por defecto está montado en todos los nodos bajo
    ``/lakehouse/default/Files``.
    """
    from pyspark.sql.types import DoubleType, StructField, StructType

    num_cols, cat_cols = list(numerical_cols), list(categorical_cols)
    schema = StructType(df.schema.fields + [StructField(output_col, DoubleType(), True)])

    def _score(iterator):
        key = (ckpt_path, tuple(num_cols), tuple(cat_cols))
        scorer = _SCORER_CACHE.get(key)
        if scorer is None:
            scorer = Scorer.from_checkpoint(ckpt_path, num_cols, cat_cols)
            _SCORER_CACHE[key] = scorer
        for pdf in iterator:
            pdf = pdf.copy()
            pdf[output_col] = scorer.predict_frame(pdf, batch_size).astype("float64") if len(pdf) else []
            yield pdf

    return df.mapInPandas(_score, schema=schema)


# ── Manifiesto ───────────────────────────────────────────────────────────────

def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str | None:
    for var in ("KANREC_GIT_SHA", "GITHUB_SHA"):
        if os.environ.get(var):
            return os.environ[var]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def write_manifest(ckpt_path: str, numerical_cols: list[str], categorical_cols: list[str],
                   scaler_stats_path: str | None = None, extra: dict | None = None,
                   manifest_path: str | None = None) -> str:
    """
    Escribe ``<ckpt>.manifest.json`` junto al checkpoint. Devuelve la ruta.

    ``extra`` admite cualquier metadato de la ejecución (seed, lr, métricas
    de test, run_id de MLflow…). No sustituye al tracking de MLflow: contiene
    lo que el servicio necesita para verificar modelo y normalización sin
    acceso al tracking server.
    """
    import torch
    from . import __version__

    state = torch.load(ckpt_path, map_location="cpu")
    arch = infer_architecture(state)
    if arch["num_numerical"] != len(numerical_cols):
        raise ValueError(
            f"el checkpoint tiene {arch['num_numerical']} campos numéricos y se "
            f"pasaron {len(numerical_cols)} nombres"
        )
    if len(arch["cat_cardinalities"]) != len(categorical_cols):
        raise ValueError("número de columnas categóricas distinto del del checkpoint")

    manifest = {
        "kanrec_version": __version__,
        "torch_version": torch.__version__,
        "git_sha": _git_sha(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": os.path.basename(ckpt_path),
        "checkpoint_sha256": sha256_of(ckpt_path),
        "scaler_stats_sha256": sha256_of(scaler_stats_path) if scaler_stats_path else None,
        "numerical_cols": list(numerical_cols),
        "categorical_cols": list(categorical_cols),
        **arch,
        "extra": extra or {},
    }
    manifest_path = manifest_path or f"{ckpt_path}.manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def load_manifest(ckpt_path: str) -> dict:
    with open(f"{ckpt_path}.manifest.json") as f:
        return json.load(f)


def check_manifest(ckpt_path: str, scaler_stats_path: str | None = None,
                   numerical_cols: list[str] | None = None) -> dict:
    """
    Verifica que el checkpoint y (opcionalmente) ``scaler_stats.json`` son
    los mismos ficheros que describe el manifiesto. Lanza ``RuntimeError``
    si no coinciden.
    """
    m = load_manifest(ckpt_path)
    problems = []
    if sha256_of(ckpt_path) != m["checkpoint_sha256"]:
        problems.append("el checkpoint no coincide con el hash del manifiesto")
    if scaler_stats_path and m.get("scaler_stats_sha256"):
        if sha256_of(scaler_stats_path) != m["scaler_stats_sha256"]:
            problems.append("scaler_stats.json no es el usado al entrenar este checkpoint")
    if numerical_cols is not None and list(numerical_cols) != m["numerical_cols"]:
        problems.append("las columnas numéricas no coinciden (nombre u orden)")
    if problems:
        raise RuntimeError("Manifiesto inconsistente: " + "; ".join(problems))
    return m
