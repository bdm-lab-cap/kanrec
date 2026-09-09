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
