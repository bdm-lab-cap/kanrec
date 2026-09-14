"""
Spark/Fabric utilities shared across notebooks.

Kept in a separate module (rather than duplicated in every notebook) so
that batch ingestion (01) and stream processing (06) apply the exact same
transformation — which is the whole point of fitting one MLlib Pipeline
and reusing it. pyspark is imported lazily inside each function so that
importing kanrec elsewhere (local training, tests) never requires it.
"""
from __future__ import annotations


def apply_pipeline_and_unpack(df, pipeline_model, std_cols):
    """
    Applies a fitted MLlib PipelineModel and unpacks StandardScaler's
    vector output back into the original scalar column names.

    Fix applied (auditoría de tribunal — hallazgo A2): StandardScaler
    writes its output into a *new* vector column (conventionally named
    "num_scaled"). If nothing unpacks it, the original scalar columns
    stay in their raw, unnormalised scale, and any downstream code that
    reads those columns by name — which is exactly what every training
    notebook did — silently trains on unnormalised data despite the
    Pipeline having "fit" a scaler.

    Args:
        df:              Input Spark DataFrame (already null-imputed and
                          log-transformed where applicable).
        pipeline_model:  A fitted PipelineModel whose stages include a
                          VectorAssembler(outputCol="num_raw") followed by
                          a StandardScaler(inputCol="num_raw",
                          outputCol="num_scaled") over exactly `std_cols`,
                          in that column order.
        std_cols:        The scalar column names StandardScaler was fit
                          over, in the same order given to VectorAssembler.

    Returns:
        The transformed DataFrame with `std_cols` overwritten by their
        normalised values, and the intermediate "num_raw" / "num_scaled"
        vector columns dropped.
    """
    from pyspark.sql import functions as F
    from pyspark.ml.functions import vector_to_array

    out = pipeline_model.transform(df)
    out = out.withColumn("num_scaled_arr", vector_to_array(F.col("num_scaled")))
    for i, c in enumerate(std_cols):
        out = out.withColumn(c, F.col("num_scaled_arr")[i])
    return out.drop("num_raw", "num_scaled", "num_scaled_arr")


def random_sample(df, n_rows: int, seed: int, safety_margin: float = 3.0):
    """
    Draws an approximately-`n_rows` random sample preserving the natural
    row order / class distribution of `df`.

    Fix applied (auditoría de tribunal — hallazgo A4): `.limit(n)` is NOT
    a random sample — it returns the first `n` rows in whatever order the
    underlying files happen to be read, which for a time-ordered dataset
    like Criteo means any two `.limit()` calls at different offsets (e.g.
    filtering positives and negatives separately, then taking `.limit()`
    of each) draw from different time windows. Any field correlated with
    position in the file — including irrelevant ones — then looks like a
    near-perfect predictor of the label, and that leak affects train,
    val and test identically, so it is invisible in the metrics.

    Uses `DataFrame.sample()` (a true random sample, one Bernoulli draw
    per row) followed by a safety-margin `.limit()`, since `fraction` only
    targets the expected row count approximately.

    Args:
        df:             Source DataFrame.
        n_rows:         Target number of rows.
        seed:           Random seed (use a different one per experiment
                         seed, not the same one every time).
        safety_margin:  Oversampling factor before the final `.limit()`,
                         to make hitting `n_rows` reliable without a second
                         pass.
    """
    total = df.count()
    if n_rows >= total:
        return df
    fraction = min(1.0, (n_rows * safety_margin) / total)
    return df.sample(withReplacement=False, fraction=fraction, seed=seed).limit(n_rows)


# ── Normalización compartida batch / stream ──────────────────────────────────
#
# 01 (ingesta batch) y 06 (procesamiento del stream) deben aplicar EXACTAMENTE
# la misma transformación. Hasta ahora cada notebook llevaba su propia copia
# del código (imputación, log1p, estandarización, 26 joins de indexado). Dos
# copias iguales hoy no garantizan dos copias iguales mañana; una función
# única sí. 01 la usa con los mapas recién ajustados sobre train; 06 la usa
# con los mismos mapas leídos de la tabla `cat_index_maps` y los estadísticos
# de `scaler_stats.json`, que 01 persiste precisamente para esto.

LOG_COLS_DEFAULT = [f"I{i}" for i in range(1, 6)]
STD_COLS_DEFAULT = [f"I{i}" for i in range(6, 14)]
CATEGORICAL_COLS_DEFAULT = [f"C{i}" for i in range(1, 27)]


def impute_and_log(df, numerical_cols, log_cols):
    """
    Imputa nulos a 0.0 y toma el valor absoluto en todas las numéricas;
    aplica log1p a las de conteo. Misma decisión (y misma limitación
    conocida: "ausente" se fusiona con "vale cero") en batch y en stream.
    """
    from pyspark.sql import functions as F

    for c in numerical_cols:
        df = df.withColumn(c, F.when(F.col(c).isNull(), 0.0).otherwise(F.abs(F.col(c))))
    for c in log_cols:
        df = df.withColumn(c, F.log1p(F.greatest(F.col(c), F.lit(0.0))))
    return df


def apply_scaler(df, scaler_stats: dict, std_cols):
    """
    Estandariza `std_cols` con media y desviación de TRAIN.

    `scaler_stats` es el contenido de scaler_stats.json:
    ``{col: {"mean": m, "std": s}}``. Se acepta también el formato en
    memoria de 01, ``{col: (m, s)}``. Una desviación nula se sustituye por 1
    para no dividir por cero (columna constante en train).
    """
    from pyspark.sql import functions as F

    for c in std_cols:
        st = scaler_stats[c]
        m, s = (st["mean"], st["std"]) if isinstance(st, dict) else (st[0], st[1])
        df = df.withColumn(c, (F.col(c) - F.lit(float(m))) / F.lit(float(s) if s else 1.0))
    return df


def load_index_maps(spark, categorical_cols, table: str = "cat_index_maps") -> dict:
    """
    Reconstruye ``{col: (mapping_df, n_cats)}`` desde la tabla que 01
    escribe con columnas (column, value, idx). `mapping_df` tiene las
    columnas ``[col, f"{col}_idx"]``, el mismo formato que 01 usa en
    memoria, para que `apply_index_maps` sea idéntica en ambos sitios.
    """
    from pyspark.sql import functions as F

    all_maps = spark.read.table(table).cache()
    counts = {r["column"]: r["n"] for r in
              all_maps.groupBy("column").agg(F.count("*").alias("n")).collect()}
    out = {}
    for c in categorical_cols:
        mp = (all_maps.filter(F.col("column") == c)
                      .select(F.col("value").alias(c), F.col("idx").alias(f"{c}_idx")))
        out[c] = (mp, int(counts.get(c, 0)))
    return out


def apply_index_maps(df, index_maps: dict, categorical_cols, chunk: int = 6,
                     materialize=None):
    """
    Aplica los mapas de indexado categórico (ajustados sobre train) con un
    broadcast join por columna. Categorías no vistas y nulos van al índice
    ``n_cats`` (equivalente a handleInvalid="keep").

    `materialize(df, i) -> df` es opcional: 01 lo usa para cortar el linaje
    cada `chunk` columnas escribiendo una tabla temporal (encadenar los 26
    joins en un único plan lo aborta el motor en Fabric). En el stream, con
    lotes pequeños, no hace falta y se pasa None.
    """
    from pyspark.sql import functions as F

    for i in range(0, len(categorical_cols), chunk):
        for c in categorical_cols[i:i + chunk]:
            mapping, n_cats = index_maps[c]
            df = (df.join(F.broadcast(mapping), on=c, how="left")
                    .withColumn(f"{c}_idx",
                                F.coalesce(F.col(f"{c}_idx"), F.lit(n_cats)).cast("int")))
        if materialize is not None:
            df = materialize(df, i)
    return df


def normalise_like_train(df, scaler_stats: dict, index_maps: dict,
                         numerical_cols=None, log_cols=None, std_cols=None,
                         categorical_cols=None, materialize=None):
    """
    Transformación completa "como train", para batch (01) y stream (06):
    imputación + log1p + estandarización con estadísticos de train +
    indexado categórico con mapas de train.
    """
    numerical_cols = numerical_cols or (LOG_COLS_DEFAULT + STD_COLS_DEFAULT)
    log_cols = log_cols or LOG_COLS_DEFAULT
    std_cols = std_cols or STD_COLS_DEFAULT
    categorical_cols = categorical_cols or CATEGORICAL_COLS_DEFAULT
    df = impute_and_log(df, numerical_cols, log_cols)
    df = apply_scaler(df, scaler_stats, std_cols)
    return apply_index_maps(df, index_maps, categorical_cols, materialize=materialize)
