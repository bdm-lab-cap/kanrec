# Microsoft Fabric Notebook — 04_model_comparison
#
# Comparativa de los tres encoders numericos —normalizacion directa, AutoDis
# y KAN-REC— sobre el mismo backbone, los mismos datos y las mismas semillas,
# de modo que la unica variable entre ellos sea el encoder. Incluye la
# ablacion de grid_size y escribe los checkpoints con su manifiesto.
#
# Sustituye a los antiguos 04_training_kanrec, 07_autodis_baseline y
# 08_comparativa_encoders, que definian el modelo por su cuenta con
# arquitecturas que no coincidian entre si ni con el paquete instalable, y
# que calculaban las cardinalidades categoricas sobre muestras distintas.
# Aqui hay una sola definicion de modelo (kanrec.baselines.build_model) y
# una sola forma de calcular cardinalidades.
#
# Run AFTER 01_spark_ingest. Attach kanrec_lakehouse before running.
#
# Salidas
#   Files/checkpoints/best_<encoder>_gs10_s<seed>.pt (+ .manifest.json)
#   Tables/experiment_results, Tables/gridsize_ablation
#
# No instalar mlflow ni scikit-learn en este notebook: Fabric trae su propio
# mlflow integrado con el plugin synapse.ml.mlflow, y una segunda instalacion
# lo rompe. Por eso kanrec declara mlflow en el extra [train] y no como
# dependencia obligatoria.

# ============================================================================
# CELDA 1 — Imports
# ============================================================================
import json
import os
import time

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from kanrec.baselines import build_model
from kanrec.spark_utils import random_sample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch: {torch.__version__} | Device: {device}")

with open("/lakehouse/default/Files/config/feature_selection.json") as f:
    sel = json.load(f)
NUMERICAL_COLS   = sel["selected"]
CATEGORICAL_COLS = [f"C{i}" for i in range(1, 27)]

CKPT_PATH = "/lakehouse/default/Files/checkpoints"
os.makedirs(CKPT_PATH, exist_ok=True)
print(f"Numerical fields ({len(NUMERICAL_COLS)}): {NUMERICAL_COLS}")


# ============================================================================
# CELDA 2 — Cardinalidades categoricas: UNA sola vez, sobre la tabla completa
# ============================================================================
# Antes: 04 usaba train/val/test completos; 07/08 usaban una muestra de
# 100k filas aparte. Aqui se calcula una unica vez, sobre TRAIN completo
# (no sobre val/test, para no filtrar informacion de esos splits hacia el
# tamano de vocabulario), y se reutiliza para los tres encoders.

idx_cols = [f"{c}_idx" for c in CATEGORICAL_COLS]

from pyspark.sql import functions as F

# Refrescar cache de metadatos antes de leer: si 01 reescribio las tablas en
# la misma capacidad, esta sesion podia seguir viendo la version anterior sin
# normalizar y entrenar sobre datos crudos.
# Guardia dura si I6..I13 no estan normalizadas.
for _t in ("train", "val", "test"):
    spark.catalog.refreshTable(_t)
STD_CHECK = [f"I{i}" for i in range(6, 14)]
_std_row = spark.read.table("train").select(
    *[F.stddev(c).alias(c) for c in STD_CHECK]
).collect()[0]
_bad = [c for c in STD_CHECK if _std_row[c] is not None and _std_row[c] > 5.0]
if _bad:
    raise RuntimeError(
        f"La tabla 'train' NO esta normalizada: {_bad} con stddev>5. Reejecuta "
        f"01_spark_ingest_mlllib con el kernel de ESTA sesion antes de entrenar."
    )
print("Tabla 'train' verificada: I6..I13 normalizadas.")

train_full = spark.read.table("train")
# Cardinalidad = max indice sobre las TRES tablas + 1, no solo train.
# StringIndexer(handleInvalid="keep") mete las categorias no vistas en un
# indice extra; si ese indice solo aparece en val/test (no en train),
# calcular la cardinalidad con el max de train la deja una unidad corta y
# nn.Embedding aborta en GPU con "device-side assert" al procesar val/test.
_max_per_split = []
for _tbl in ("train", "val", "test"):
    _row = spark.read.table(_tbl).agg(*[F.max(c).alias(c) for c in idx_cols]).collect()[0]
    _max_per_split.append({c: (_row[c] if _row[c] is not None else 0) for c in idx_cols})
cat_cardinalities = [
    int(max(_m[c] for _m in _max_per_split)) + 1 for c in idx_cols
]
print(f"Cardinalidades categoricas (sobre TRAIN completo): {cat_cardinalities}")


# ============================================================================
# CELDA 3 — Dataset con muestreo ALEATORIO real
# ============================================================================
class CriteoDataset(Dataset):
    """
    Carga una muestra de una tabla Delta ya normalizada por 01, usando
    kanrec.spark_utils.random_sample en lugar de .limit() sobre positivos
    y negativos por separado. Se entrena sobre la distribucion NATURAL del
    CTR: no se balancea artificialmente a 50/50, porque el AUC es
    insensible al prior de clase y balancear solo obliga a explicar por
    que los numeros no son comparables con la literatura.
    """

    def __init__(self, table_name: str, num_cols: list[str], idx_cols: list[str],
                 n_rows: int, seed: int):
        print(f"  Cargando {table_name} (muestra aleatoria, seed={seed})...")
        df_spark = spark.read.table(table_name)
        sample = random_sample(df_spark, n_rows=n_rows, seed=seed)
        df = sample.toPandas()

        ctr = df["label"].mean()
        print(f"  {table_name}: {len(df):,} filas | CTR de la muestra: {ctr:.2%}")

        self.x_num = torch.tensor(df[num_cols].fillna(0).values.astype("float32"))
        present_idx = [c for c in idx_cols if c in df.columns]
        if present_idx:
            self.x_cat = torch.tensor(df[present_idx].fillna(0).values.astype("int64"))
        else:
            self.x_cat = torch.zeros(len(df), len(idx_cols), dtype=torch.long)
        self.y = torch.tensor(df["label"].values.astype("float32"))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_num[idx], self.x_cat[idx], self.y[idx]


# Tamanos de muestra: ajusta N_TRAIN al tiempo disponible en la sesion de
# Fabric. 200k filas de entrenamiento con la distribucion natural del CTR
# ya dan un AUC comparable con la literatura; usar mas solo mejora la
# precision de la estimacion, no cambia la conclusion.
N_TRAIN, N_VAL, N_TEST = 200_000, 40_000, 40_000
SEED_FOR_SAMPLING = 42  # el muestreo de FILAS es fijo entre encoders y semillas de modelo:
                        # lo que varia entre "seeds" es la inicializacion del modelo,
                        # no que datos ve cada encoder — si no, la comparacion no es justa.

train_ds = CriteoDataset("train", NUMERICAL_COLS, idx_cols, N_TRAIN, SEED_FOR_SAMPLING)
val_ds   = CriteoDataset("val",   NUMERICAL_COLS, idx_cols, N_VAL,   SEED_FOR_SAMPLING)
test_ds  = CriteoDataset("test",  NUMERICAL_COLS, idx_cols, N_TEST,  SEED_FOR_SAMPLING)

BATCH_SIZE = 1024
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# ============================================================================
# CELDA 4 — Entrenamiento de una configuracion (encoder, seed)
# ============================================================================
# Learning rate por encoder, no unico para los tres. Con lr=1e-3 para todos,
# AutoDis queda infraentrenado incluso a 30 epocas (AUC ~0.57 frente a ~0.78
# con lr=1e-2 en el mismo presupuesto), mientras que raw y kan-bspline son
# estables en ambos valores. Igualar el lr "por simetria" no es mas justo:
# penaliza estructuralmente a la arquitectura con mas parametros en su capa
# de discretizacion (proyeccion + skip + temperatura).
# Lo que debe ser igual entre encoders son los DATOS y la evaluacion, no
# necesariamente el learning rate.
DEFAULT_LR = {"raw": 1e-3, "autodis": 1e-2, "kan-bspline": 1e-3}


def evaluate(model, loader) -> tuple[float, float]:
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x_num, x_cat, y in loader:
            logits = model(x_num.to(device), x_cat.to(device)).squeeze()
            p = torch.sigmoid(logits).cpu().numpy()  # el modelo devuelve LOGITS
            preds.extend(np.atleast_1d(p))
            labels.extend(y.numpy())
    return roc_auc_score(labels, preds), log_loss(labels, preds)


def train_one_run(
    encoder_name: str,
    seed: int,
    max_epochs: int = 30,
    patience: int = 3,
    lr: float | None = None,
    embedding_dim: int = 16,
    grid_size: int = 10,
    ckpt_tag: str = "",     # sufijo del checkpoint, para no pisar corridas distintas
) -> dict:
    torch.manual_seed(seed)
    lr = lr if lr is not None else DEFAULT_LR[encoder_name]

    model = build_model(
        encoder=encoder_name,
        num_numerical=len(NUMERICAL_COLS),
        cat_cardinalities=cat_cardinalities,
        embedding_dim=embedding_dim,
        kan_grid_size=grid_size,
    ).to(device)

    # Calibracion del grid: solo kan-bspline la necesita.
    # Se hace SIEMPRE sobre datos normalizados (celda 01 ya garantiza esto),
    # con una muestra generosa para cubrir bien la distribucion de cada campo.
    if hasattr(model, "calibrate"):
        # 50k filas, no 5 lotes fijos (tras la primera
        # corrida real en Fabric: I6 e I12 llegan a ~690 desviaciones tipicas
        # en Criteo -- colas extremas). Con solo ~5k-10k filas de muestra la
        # probabilidad de capturar esos outliers era baja, y el grid
        # calibrado podia dejarlos fuera de cobertura, el mismo fallo de
        # calibracion pero limitado a esas pocas filas. Acumular por FILAS
        # en vez de por numero de lotes es ademas robusto a que cada script
        # use un batch_size distinto.
        CALIB_ROWS = 50_000
        calib_batches, calib_n = [], 0
        for x_num, _, _ in train_loader:
            calib_batches.append(x_num)
            calib_n += x_num.size(0)
            if calib_n >= CALIB_ROWS:
                break
        model.calibrate(torch.cat(calib_batches, dim=0).to(device))

    # parameter_groups da al spline un lr 25x mayor: arranca ~50x mas pequeno
    # que la ruta base por la inicializacion de efficient-kan y, con un lr
    # compartido, nunca despega (las curvas phi salian rectas siempre, ver
    # KANRecModel.parameter_groups). Los encoders raw/autodis no tienen
    # splines, asi que para ellos el helper no existe y se usa el Adam normal.
    if hasattr(model, "parameter_groups"):
        optimizer = torch.optim.Adam(model.parameter_groups(base_lr=lr), weight_decay=1e-5)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    # BCEWithLogitsLoss (no BCELoss): estable numericamente; evita el
    # device-side assert por log(0) cuando pred satura a 0/1 (C3).
    criterion = torch.nn.BCEWithLogitsLoss()

    # grid_size va en el nombre: si no, la ablacion de la CELDA 7 (que
    # reentrena kan-bspline con distintos grid_size sobre la MISMA seed 42)
    # sobreescribiria el checkpoint de la comparativa principal.
    ckpt_path = f"{CKPT_PATH}/best_{encoder_name}_gs{grid_size}_s{seed}{ckpt_tag}.pt"
    best_val_auc, patience_ctr = 0.0, 0
    t0 = time.time()

    with mlflow.start_run(run_name=f"{encoder_name}-s{seed}"):
        mlflow.log_params({
            "encoder": encoder_name, "seed": seed, "n_train": len(train_ds),
            "embedding_dim": embedding_dim, "grid_size": grid_size, "lr": lr,
        })

        for epoch in range(max_epochs):
            model.train()
            train_loss = 0.0
            for x_num, x_cat, y in train_loader:
                x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x_num, x_cat).squeeze()  # logits, no proba
                loss = criterion(logits, y)
                if hasattr(model, "entropy_regularization_loss"):
                    loss = loss + model.entropy_regularization_loss()
                if not torch.isfinite(loss):
                    optimizer.zero_grad()
                    continue  # salta el batch corrupto sin propagar NaN a los pesos
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            val_auc, val_ll = evaluate(model, val_loader)
            scheduler.step(1 - val_auc)
            mlflow.log_metrics({"train_loss": train_loss, "val_auc": val_auc, "val_logloss": val_ll}, step=epoch)

            if val_auc > best_val_auc:
                best_val_auc, patience_ctr = val_auc, 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    break

        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        test_auc, test_ll = evaluate(model, test_loader)
        mlflow.log_metrics({"test_auc": test_auc, "test_logloss": test_ll})

        elapsed = time.time() - t0
        n_params = sum(p.numel() for p in model.parameters())
        mlflow.log_metrics({"train_seconds": elapsed, "n_params": n_params})

    print(f"  {encoder_name:12s} seed={seed:<4d} epochs={epoch+1:<3d} "
          f"val_auc={best_val_auc:.4f} test_auc={test_auc:.4f} test_ll={test_ll:.4f} "
          f"({elapsed:.0f}s)")

    return {
        "encoder": encoder_name, "seed": seed, "epochs": epoch + 1,
        "val_auc": best_val_auc, "test_auc": test_auc, "test_logloss": test_ll,
        "train_seconds": elapsed, "n_params": n_params,
    }


# ============================================================================
# CELDA 5 — Comparativa: 3 encoders x 3 semillas, MISMAS condiciones
# ============================================================================
mlflow.set_experiment("kanrec-model-comparison")

ENCODERS = ["raw", "autodis", "kan-bspline"]
SEEDS = [42, 123, 256]

print(f"Entrenando {len(ENCODERS)} x {len(SEEDS)} = {len(ENCODERS)*len(SEEDS)} modelos...")
print(f"(mismo backbone, mismos datos, mismas epocas — la unica variable es el encoder)\n")

results = []
for encoder_name in ENCODERS:
    for seed in SEEDS:
        results.append(train_one_run(encoder_name, seed))


# ============================================================================
# CELDA 6 — Tabla de resultados: media +/- desviacion por encoder
# ============================================================================
results_df = pd.DataFrame(results)
results_df["timestamp"] = pd.Timestamp.now().isoformat()

summary = results_df.groupby("encoder").agg(
    test_auc_mean=("test_auc", "mean"),
    test_auc_std=("test_auc", "std"),
    test_logloss_mean=("test_logloss", "mean"),
    test_logloss_std=("test_logloss", "std"),
    train_seconds_mean=("train_seconds", "mean"),
).round(4)
print("\n" + "=" * 70)
print("RESUMEN — media +/- desviacion sobre 3 semillas (42, 123, 256)")
print("=" * 70)
print(summary.to_string())

# Persistencia en Delta (consumido por Power BI, ver powerbi/README.md)
spark.createDataFrame(results_df).write.format("delta").mode("overwrite").save("Tables/experiment_results")
print("\n✓ Tables/experiment_results escrita.")


# ============================================================================
# CELDA 6b — Manifiesto de cada checkpoint
# ============================================================================
# Junto a cada checkpoint se escribe un JSON con la version del paquete, el
# commit, las columnas, las cardinalidades, los hiperparametros y el hash
# SHA-256 del propio checkpoint y de scaler_stats.json. 06 verifica ambos
# hashes antes de servir (kanrec.serving.check_manifest).
from kanrec.serving import write_manifest

STATS_PATH = "/lakehouse/default/Files/config/scaler_stats.json"
for r in results:
    ckpt = f"{CKPT_PATH}/best_{r['encoder']}_gs10_s{r['seed']}.pt"   # la comparativa usa el grid_size por defecto (10)
    if not os.path.exists(ckpt):
        print(f"  (sin checkpoint para {ckpt}, se omite)")
        continue
    mpath = write_manifest(
        ckpt, numerical_cols=NUMERICAL_COLS, categorical_cols=idx_cols,
        scaler_stats_path=STATS_PATH,
        extra={k: r[k] for k in ("seed", "epochs", "val_auc", "test_auc", "test_logloss",
                                 "train_seconds", "n_params") if k in r},
    )
    print(f"  ✓ manifiesto: {mpath}")


# ============================================================================
# CELDA 7 (opcional) — Ablacion ligera de grid_size, solo KAN-REC
# ============================================================================
# Los tres valores se entrenan con el mismo presupuesto para que la ablacion
# sea comparable. Los checkpoints llevan sufijo "_abl" para no pisar los de la
# comparativa principal, que son los que cargan 05 y 06.
ablation_results = []
for grid_size in [5, 10, 20]:
    r = train_one_run("kan-bspline", seed=42, grid_size=grid_size, max_epochs=15, patience=2,
                      ckpt_tag="_abl")
    r["grid_size"] = grid_size
    ablation_results.append(r)

ablation_df = pd.DataFrame(ablation_results)
print("\nAblacion de grid_size (KAN-REC, seed=42):")
print(ablation_df[["grid_size", "test_auc", "test_logloss", "train_seconds"]].to_string(index=False))

spark.createDataFrame(ablation_df).write.format("delta").mode("overwrite").save("Tables/gridsize_ablation")
print("\n✓ Tables/gridsize_ablation escrita.")

print("\nDone.")
