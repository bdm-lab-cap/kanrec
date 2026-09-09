# Microsoft Fabric Notebook — 04_model_comparison
# SUSTITUYE a los antiguos 04_training_kanrec, 07_autodis_baseline y
# 08_comparativa_encoders, que quedan eliminados del repositorio.
#
# Por que se consolidan los tres en uno (auditoria de tribunal - hallazgo B5/B6):
#   - Los tres definian KANRecModel por su cuenta, con arquitecturas que no
#     coincidian entre si ni con el paquete instalable.
#   - 04 calculaba las cardinalidades categoricas sobre train/val/test
#     completos; 07 y 08 las calculaban sobre una muestra de 100k filas
#     distinta. Los tres encoders no veian la misma resolucion categorica.
#   - train.py --encoder no tenia efecto real: las tres ramas construian
#     KANRecModel sin importar el flag.
# Aqui hay UNA sola definicion de modelo (kanrec.baselines.build_model),
# UNA sola forma de calcular cardinalidades, y los tres encoders comparten
# exactamente el mismo backbone de interaccion — asi que la unica variable
# entre ellos es, de verdad, el encoder numerico.
#
# Fixes aplicados en este notebook:
#   A2  — lee Tables/train ya normalizado por 01 (StandardScaler llega
#         de verdad a I6..I13).
#   A3  — calibra el grid de las B-splines a la distribucion real de cada
#         campo antes de entrenar (kanrec.encoder.calibrate).
#   A4  — muestreo aleatorio real (kanrec.spark_utils.random_sample) en
#         vez de .limit() por separado sobre positivos/negativos, que no
#         es un muestreo aleatorio y contaminaba con senal temporal.
#         Se abandona el balanceo artificial 50/50: se entrena sobre la
#         distribucion natural del CTR, que es lo que hace comparable el
#         AUC con la literatura publicada (DeepFM, DCNv2, AutoInt ~0.80-0.815).
#   A5  — AutoDis sin el sigmoid antes del softmax (kanrec.baselines).
#   A6  — regularizacion de entropia > 0 de verdad (kanrec.encoder).
#
# Requiere el paquete kanrec instalado en esta sesion:
#   %pip install --quiet "git+https://github.com/bdm-lab-cap/kanrec.git@main"

# ============================================================================
# CELDA 1 — Instalacion e imports
# ============================================================================
# %pip install --quiet "git+https://github.com/bdm-lab-cap/kanrec.git@main"
# %pip install --quiet scikit-learn mlflow

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
train_full = spark.read.table("train")

from pyspark.sql import functions as F
cat_max_row = train_full.agg(*[F.max(c).alias(c) for c in idx_cols]).collect()[0]
cat_cardinalities = [int(cat_max_row[c]) + 1 for c in idx_cols]
print(f"Cardinalidades categoricas (sobre TRAIN completo): {cat_cardinalities}")


# ============================================================================
# CELDA 3 — Dataset con muestreo ALEATORIO real (hallazgo A4)
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
def evaluate(model, loader) -> tuple[float, float]:
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x_num, x_cat, y in loader:
            p = model(x_num.to(device), x_cat.to(device)).squeeze().cpu().numpy()
            preds.extend(np.atleast_1d(p))
            labels.extend(y.numpy())
    return roc_auc_score(labels, preds), log_loss(labels, preds)


def train_one_run(
    encoder_name: str,
    seed: int,
    max_epochs: int = 30,
    patience: int = 3,
    lr: float = 1e-3,
    embedding_dim: int = 16,
    grid_size: int = 10,
) -> dict:
    torch.manual_seed(seed)

    model = build_model(
        encoder=encoder_name,
        num_numerical=len(NUMERICAL_COLS),
        cat_cardinalities=cat_cardinalities,
        embedding_dim=embedding_dim,
        kan_grid_size=grid_size,
    ).to(device)

    # Calibracion del grid (hallazgo A3): solo kan-bspline la necesita.
    # Se hace SIEMPRE sobre datos normalizados (celda 01 ya garantiza esto),
    # con una muestra generosa para cubrir bien la distribucion de cada campo.
    if hasattr(model, "calibrate"):
        calib_batches = []
        for i, (x_num, _, _) in enumerate(train_loader):
            calib_batches.append(x_num)
            if i >= 4:
                break
        model.calibrate(torch.cat(calib_batches, dim=0).to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    criterion = torch.nn.BCELoss()

    # grid_size va en el nombre: si no, la ablacion de la CELDA 7 (que
    # reentrena kan-bspline con distintos grid_size sobre la MISMA seed 42)
    # sobreescribiria el checkpoint de la comparativa principal.
    ckpt_path = f"{CKPT_PATH}/best_{encoder_name}_gs{grid_size}_s{seed}.pt"
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
                y_pred = model(x_num, x_cat).squeeze()
                loss = criterion(y_pred, y)
                if hasattr(model, "entropy_regularization_loss"):
                    loss = loss + model.entropy_regularization_loss()
                loss.backward()
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
print("\nNOTA: si test_auc_std es del mismo orden que la diferencia entre dos")
print("encoders, esa diferencia NO es significativa. Dilo asi en la memoria")
print("en vez de quedarte solo con la media.")

# Persistencia en Delta (consumido por Power BI, ver powerbi/README.md)
spark.createDataFrame(results_df).write.format("delta").mode("overwrite").save("Tables/experiment_results")
print("\n✓ Tables/experiment_results escrita.")


# ============================================================================
# CELDA 7 (opcional) — Ablacion ligera de grid_size, solo KAN-REC
# ============================================================================
# Un resultado adicional barato: si grid_size=5 (el valor por defecto de
# efficient-kan) es notablemente peor que 10 o 20 ahora que el grid SI se
# calibra, es evidencia de que la capacidad del spline importa una vez que
# el rango es el correcto — que es justo lo que la memoria puede reportar
# como "Resultado 2" ademas de la comparativa principal.
ablation_results = []
for grid_size in [5, 10, 20]:
    r = train_one_run("kan-bspline", seed=42, grid_size=grid_size, max_epochs=15, patience=2)
    r["grid_size"] = grid_size
    ablation_results.append(r)

ablation_df = pd.DataFrame(ablation_results)
print("\nAblacion de grid_size (KAN-REC, seed=42):")
print(ablation_df[["grid_size", "test_auc", "test_logloss", "train_seconds"]].to_string(index=False))

spark.createDataFrame(ablation_df).write.format("delta").mode("overwrite").save("Tables/gridsize_ablation")
print("\n✓ Tables/gridsize_ablation escrita.")

print("\nDone.")
