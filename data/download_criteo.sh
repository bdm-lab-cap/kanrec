#!/bin/bash
# Descarga el dataset Criteo Display Advertising y genera la submuestra de 10M filas.
#
# NOTA sobre la fuente (verificado en 2026): la competicion de Kaggle
# "criteo-display-ad-challenge" ya no permite la descarga del fichero, asi que
# el script original (`kaggle competitions download -c ...`) no funcionaba y la
# ingesta no era reproducible por nadie. Se usan en su lugar los mirrors
# publicos de Criteo AI Lab y Hugging Face.
#
# El fichero completo son ~4,4 GB comprimidos / ~11 GB sin comprimir. La
# submuestra de 10M filas que usa el proyecto ocupa 2,26 GB.
set -e

DATA_DIR="${1:-data}"
mkdir -p "$DATA_DIR"

CRITEO_URL="https://huggingface.co/datasets/criteo/CriteoClickLogs/resolve/main/day_0.gz"
FALLBACK_URL="https://storage.googleapis.com/criteo-cail-datasets/dac.tar.gz"

echo "=== Descargando Criteo ==="
if [ -f "$DATA_DIR/criteo_10m.tsv" ]; then
    echo "Ya existe $DATA_DIR/criteo_10m.tsv ($(du -sh "$DATA_DIR/criteo_10m.tsv" | cut -f1)). Nada que hacer."
    exit 0
fi

if ! curl -fsIL "$CRITEO_URL" >/dev/null 2>&1; then
    echo "Mirror principal no disponible; probando el de Criteo AI Lab..."
    CRITEO_URL="$FALLBACK_URL"
fi

curl -fL --progress-bar "$CRITEO_URL" -o "$DATA_DIR/criteo_raw.gz"

echo "=== Extrayendo las primeras 10.000.001 filas ==="
# Se extrae en streaming para no descomprimir los 11 GB completos.
gunzip -c "$DATA_DIR/criteo_raw.gz" | head -n 10000001 > "$DATA_DIR/criteo_10m.tsv"

N_LINES=$(wc -l < "$DATA_DIR/criteo_10m.tsv")
N_COLS=$(head -1 "$DATA_DIR/criteo_10m.tsv" | awk -F'\t' '{print NF}')

echo ""
echo "=== Verificacion ==="
echo "  filas:    $N_LINES  (esperado 10000001)"
echo "  columnas: $N_COLS   (esperado 40: 1 label + 13 numericas + 26 categoricas)"

if [ "$N_LINES" -ne 10000001 ] || [ "$N_COLS" -ne 40 ]; then
    echo "ERROR: el fichero no tiene la forma esperada. Revisa la fuente de descarga."
    exit 1
fi

# El MD5 del fichero completo no se comprueba porque los mirrors sirven
# versiones con distinta compresion; se verifica la FORMA de la submuestra,
# que es lo que consume el pipeline.
echo "  OK: $DATA_DIR/criteo_10m.tsv ($(du -sh "$DATA_DIR/criteo_10m.tsv" | cut -f1))"
echo ""
echo "Siguiente paso: subir el fichero a Files/raw/ del lakehouse de Fabric,"
echo "o a Google Drive si se va a ejecutar el notebook de Colab."
