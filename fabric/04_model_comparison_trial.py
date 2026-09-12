# Microsoft Fabric Notebook — 04_model_comparison_TRIAL
#
# Version ajustada a las limitaciones de la capacidad TRIAL de Fabric.
# NO es la fuente de las cifras finales de la memoria: es la prueba de que
# el pipeline completo (Spark MLlib -> KAN-REC -> MLflow -> Delta) funciona
# de principio a fin en la plataforma real, con datos y entrenamiento reales
# (no un mock), a una escala que cabe con seguridad en la trial.
#
# La comparativa ESTADISTICAMENTE ROBUSTA (3 encoders x 3 semillas + ablacion
# de grid_size) se ejecuta en Google Colab Pro sobre la MISMA muestra que
# exporta la Celda 4 de 01_spark_ingest_mlllib — ver colab/kanrec_full_comparison.ipynb.
#
# Por que se reduce el alcance aqui, con datos concretos:
#   - La capacidad trial NO admite cola de trabajos: un pico de uso se
#     rechaza al momento con 430 TooManyRequestsForCapacity, no espera
#     (a diferencia de una capacidad de pago). No hay margen para que un
#     bucle largo se recupere de un fallo transitorio.
#   - La sesion de Spark se desaloja tras 20 min de inactividad, y el
#     arranque en frio tarda 3-8 min. Un bucle desatendido de 9
#     entrenamientos + ablacion es exactamente el tipo de ejecucion larga
#     que puede chocar con esto.
#   - La documentacion de la comunidad de Fabric recomienda explicitamente,
#     para capacidades trial, usar un pool PEQUENO en vez del Starter Pool
#     por defecto.
#
# Que SI demuestra este notebook, y que cuenta para el criterio 1 del
# concurso (aplicacion de tecnicas del master):
#   - El pipeline de Spark MLlib (01) alimentando de verdad al entrenamiento.
#   - Las 3 arquitecturas (raw/AutoDis/KAN-REC) entrenando y evaluando en
#     Fabric mismo, con MLflow registrando los runs.
#   - Una tabla experiment_results real en Delta, consumible por Power BI.
#   - Higiene de sesion explicita (spark.stop() al final), que es la
#     practica recomendada precisamente para no agotar el cupo del trial.
#
# Requiere el paquete kanrec instalado en esta sesion, FIJADO al mismo
# commit que usa el notebook de Colab (para que ambos entornos ejecuten
# exactamente el mismo codigo):
#   %pip install --quiet "git+https://github.com/bdm-lab-cap/kanrec.git@main"
#
# IMPORTANTE (fallo real, verificado en Fabric el 2026-09-09): NO instales
# mlflow ni scikit-learn por separado aqui. Fabric ya trae un mlflow
# propio, integrado con su plugin synapse.ml.mlflow. Si se instala OTRO
# mlflow (via pip, o transitivamente al instalar kanrec en versiones
# anteriores a la 0.3.2), el plugin de Fabric se rompe con:
#   "wrapper() got an unexpected keyword argument 'expected_status'"
# kanrec>=0.3.2 ya NO trae mlflow como dependencia obligatoria por esto
# mismo. scikit-learn si viene con kanrec, no hace falta instalarlo aparte.
# (Se usa @main temporalmente: repinear al commit exacto tras hacer
#  push de este arreglo -- ver "Pasos a seguir".)

# ============================================================================
# CELDA 1 — Configuracion de sesion + instalacion + imports
# ============================================================================
# Antes de nada: usa un pool PEQUENO, no el Starter Pool por defecto.
# En el workspace: Configuracion -> Data Engineering/Science -> Spark
# Settings -> Pool -> elige un pool "Small" (o crea uno de 4 vCores).
# Esto es lo que la comunidad de Fabric recomienda para capacidades trial.

# %pip install --quiet "git+https://github.com/bdm-lab-cap/kanrec.git@main"
# NO instales mlflow ni scikit-learn aqui -- ver nota arriba. Fabric ya
# los trae, y reinstalarlos rompe el plugin de mlflow propio de Fabric.

# ---------------------------------------------------------------------------
# IMPORTANTE — instalacion de kanrec en ejecucion por PIPELINE
#
# `%pip install` esta DESHABILITADO cuando un notebook se ejecuta desde un
# Data Pipeline: solo funciona en sesiones interactivas. Verificado en Fabric:
#   MagicUsageError: %pip magic command is disabled
#
# Por eso la primera celda de cada notebook NO instala nada. El paquete se
# resuelve por una de estas dos vias, ambas compatibles con pipeline:
#
#   A) Carpeta en Files (rapida, sin publicar entorno). Subir la carpeta
#      `kanrec/` a Files/libs/ y anadir al inicio del notebook:
#
#          import sys
#          sys.path.insert(0, "/lakehouse/default/Files/libs")
#
#   B) Entorno de Fabric (la via formal). Workspace -> Nuevo -> Entorno ->
#      Bibliotecas personalizadas -> subir kanrec-0.3.2-py3-none-any.whl ->
#      Publicar -> asignar el entorno al workspace o al notebook.
#
# Las dependencias (torch, scipy, scikit-learn, pandas, pyarrow) ya vienen en
# el runtime de Fabric, asi que ninguna de las dos vias necesita resolverlas.
# ---------------------------------------------------------------------------
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
if device.type == "cpu":
    print("Aviso: sin GPU en esta sesion (esperado en la trial). Por eso "
          "esta version usa una escala reducida; la comparativa completa "
          "va en Colab Pro.")

with open("/lakehouse/default/Files/config/feature_selection.json") as f:
    sel = json.load(f)
NUMERICAL_COLS   = sel["selected"]
CATEGORICAL_COLS = [f"C{i}" for i in range(1, 27)]

CKPT_PATH = "/lakehouse/default/Files/checkpoints"
os.makedirs(CKPT_PATH, exist_ok=True)


# ============================================================================
# CELDA 2 — Cardinalidades categoricas (igual que en la version completa)
# ============================================================================
from pyspark.sql import functions as F

# Refrescar la cache de metadatos de Spark ANTES de leer nada. Sin esto, si
# 01 reescribio las tablas ('overwrite') en la misma capacidad de Fabric,
# esta sesion podia seguir viendo el esquema/datos anteriores en cache y
# entrenar sobre la tabla vieja SIN normalizar (rango I6 hasta ~12000 en vez
# de ~[-3,3]), produciendo curvas espuriamente lineales. Verificado: era la
# causa de que 04_trial diera el mismo AUC pese a haber reejecutado 01.
for _t in ("train", "val", "test"):
    spark.catalog.refreshTable(_t)

# Guardia: si I6..I13 no estan normalizadas en la tabla que vamos a leer,
# parar aqui en vez de entrenar sobre datos crudos. Reejecuta 01 primero.
STD_CHECK = [f"I{i}" for i in range(6, 14)]
_std_row = spark.read.table("train").select(
    *[F.stddev(c).alias(c) for c in STD_CHECK]
).collect()[0]
_bad = [c for c in STD_CHECK if _std_row[c] is not None and _std_row[c] > 5.0]
if _bad:
    raise RuntimeError(
        f"La tabla 'train' NO esta normalizada: {_bad} con stddev>5 (deberia "
        f"ser ~1.0). Reejecuta 01_spark_ingest_mlllib (con el kernel de ESTA "
        f"sesion, para que el refresh de cache tenga efecto) antes de entrenar."
    )
print("Tabla 'train' verificada: I6..I13 normalizadas.")

idx_cols = [f"{c}_idx" for c in CATEGORICAL_COLS]
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
print(f"Cardinalidades categoricas: {cat_cardinalities}")

# Liberamos train_full explicitamente: en un pool pequeno, cachear un
# DataFrame grande sin necesidad es la forma mas facil de agotar memoria.
train_full.unpersist()
del train_full


# ============================================================================
# CELDA 3 — Dataset REDUCIDO: suficiente para probar el pipeline, no para
# producir cifras finales (esas salen de Colab, con mas filas y mas GPU).
# ============================================================================
class CriteoDatasetSmall(Dataset):
    def __init__(self, table_name: str, num_cols: list[str], idx_cols: list[str],
                 n_rows: int, seed: int):
        print(f"  Cargando {table_name} (n={n_rows:,}, seed={seed})...")
        sample = random_sample(spark.read.table(table_name), n_rows=n_rows, seed=seed)
        df = sample.toPandas()
        print(f"  {table_name}: {len(df):,} filas | CTR de la muestra: {df['label'].mean():.2%}")

        self.x_num = torch.tensor(df[num_cols].fillna(0).values.astype("float32"))
        present_idx = [c for c in idx_cols if c in df.columns]
        self.x_cat = torch.tensor(df[present_idx].fillna(0).values.astype("int64")) \
            if present_idx else torch.zeros(len(df), len(idx_cols), dtype=torch.long)
        self.y = torch.tensor(df["label"].values.astype("float32"))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_num[idx], self.x_cat[idx], self.y[idx]


# Escala deliberadamente pequena: suficiente para que el pipeline se
# ejercite de verdad y las metricas tengan sentido, sin arriesgar la
# sesion. Sube estos numeros solo si tu capacidad no es la trial.
N_TRAIN, N_VAL, N_TEST = 40_000, 8_000, 8_000
SEED_FOR_SAMPLING = 42

train_ds = CriteoDatasetSmall("train", NUMERICAL_COLS, idx_cols, N_TRAIN, SEED_FOR_SAMPLING)
val_ds   = CriteoDatasetSmall("val",   NUMERICAL_COLS, idx_cols, N_VAL,   SEED_FOR_SAMPLING)
test_ds  = CriteoDatasetSmall("test",  NUMERICAL_COLS, idx_cols, N_TEST,  SEED_FOR_SAMPLING)

BATCH_SIZE = 512
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# ============================================================================
# CELDA 4 — Entrenamiento de una configuracion (igual logica que la version
# completa; menos epocas y patience mas corto para acotar el tiempo total)
# ============================================================================
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


# Learning rate por encoder (mismo criterio que la version completa de
# Colab): con lr=1e-3 igual para los tres, AutoDis queda infraentrenado
# incluso a 30 epocas por su capa de discretizacion mas parametrizada.
# Verificado empiricamente antes de entregar este notebook.
DEFAULT_LR = {"raw": 1e-3, "autodis": 1e-2, "kan-bspline": 1e-3}


def train_one_run(encoder_name: str, seed: int, max_epochs: int = 12,
                   patience: int = 2, lr: float | None = None,
                   embedding_dim: int = 16, grid_size: int = 10) -> dict:
    torch.manual_seed(seed)
    lr = lr if lr is not None else DEFAULT_LR[encoder_name]

    model = build_model(
        encoder=encoder_name, num_numerical=len(NUMERICAL_COLS),
        cat_cardinalities=cat_cardinalities, embedding_dim=embedding_dim,
        kan_grid_size=grid_size,
    ).to(device)

    if hasattr(model, "calibrate"):
        # 20k filas (escala trial reducida), no 3 lotes fijos -- mismo
        # motivo que en fabric/04_model_comparison.py: colas extremas en
        # I6-I13 (~690 desviaciones tipicas) que una muestra muy pequena
        # podia dejar fuera del grid calibrado.
        CALIB_ROWS = 20_000
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.5)
    # BCEWithLogitsLoss (no BCELoss): estable numericamente; evita el
    # device-side assert por log(0) cuando pred satura a 0/1 (C3).
    criterion = torch.nn.BCEWithLogitsLoss()

    ckpt_path = f"{CKPT_PATH}/trial_{encoder_name}_gs{grid_size}_s{seed}.pt"
    best_val_auc, patience_ctr = 0.0, 0
    t0 = time.time()

    with mlflow.start_run(run_name=f"TRIAL-{encoder_name}-s{seed}"):
        mlflow.log_params({"encoder": encoder_name, "seed": seed, "n_train": len(train_ds),
                            "embedding_dim": embedding_dim, "grid_size": grid_size,
                            "lr": lr, "scope": "fabric-trial-smoke-test"})

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
        elapsed = time.time() - t0
        mlflow.log_metrics({"test_auc": test_auc, "test_logloss": test_ll, "train_seconds": elapsed})

    print(f"  {encoder_name:12s} seed={seed:<4d} epochs={epoch+1:<3d} "
          f"val_auc={best_val_auc:.4f} test_auc={test_auc:.4f} ({elapsed:.0f}s)")
    return {"encoder": encoder_name, "seed": seed, "epochs": epoch + 1,
            "val_auc": best_val_auc, "test_auc": test_auc, "test_logloss": test_ll,
            "train_seconds": elapsed}


# ============================================================================
# CELDA 5 — Comparativa REDUCIDA: 3 encoders, 1 sola semilla
# ============================================================================
# Una sola semilla es intencional: esto NO reemplaza la comparativa de 3
# semillas de Colab, solo demuestra que el pipeline funciona en Fabric con
# los 3 encoders reales. No reportes esto como el resultado principal en la
# memoria — reportalo como "prueba de funcionamiento end-to-end en Fabric";
# el resultado principal, con barras de error entre semillas, viene de Colab.
mlflow.set_experiment("kanrec-model-comparison-TRIAL")

ENCODERS = ["raw", "autodis", "kan-bspline"]
SEED = 42

print(f"Entrenando {len(ENCODERS)} modelos (1 semilla, escala reducida)...\n")
results = [train_one_run(enc, SEED) for enc in ENCODERS]


# ============================================================================
# CELDA 6 — Persistencia en Delta + higiene de sesion
# ============================================================================
results_df = pd.DataFrame(results)
results_df["timestamp"] = pd.Timestamp.now().isoformat()
results_df["scope"] = "fabric-trial-smoke-test"

print("\n" + "=" * 60)
print("RESULTADOS (escala reducida, 1 semilla — prueba de pipeline)")
print("=" * 60)
print(results_df[["encoder", "test_auc", "test_logloss", "train_seconds"]].to_string(index=False))
print("\nLas cifras que van en la memoria salen de Colab (3 semillas, mas "
      "filas, GPU). Esto solo demuestra que el pipeline de Fabric produce "
      "modelos reales y coherentes.")

spark.createDataFrame(results_df).write.format("delta").mode("overwrite").save("Tables/experiment_results_trial")
print("\n✓ Tables/experiment_results_trial escrita.")

# Cierre explicito de la sesion: libera el cupo de la capacidad trial en
# vez de dejar la sesion "zombi" consumiendo VCores hasta el timeout de
# 20 min. Es la practica que recomienda la documentacion de Fabric.
try:
    mssparkutils.session.stop()
except NameError:
    spark.stop()
print("Sesion de Spark cerrada explicitamente.")
