#!/bin/bash
# Empaqueta el entregable final del TFM segun la guia del master.
#
# Produce Pedro_Martinez_Sanchez_TFM.zip con la estructura que pide la guia:
# memoria, anexos, codigo fuente y referencia al repositorio.
#
# Uso:  bash scripts/empaquetar_entrega.sh
set -e

NOMBRE="Pedro_Martinez_Sanchez_TFM"
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
SALIDA="$RAIZ/entrega"
REPO_URL="https://github.com/bdm-lab-cap/kanrec"

echo "=== Preparando $NOMBRE ==="
rm -rf "$SALIDA" "$RAIZ/$NOMBRE.zip"
mkdir -p "$SALIDA"/{memoria,codigo,resultados}

# ── 1. Memoria y anexos ────────────────────────────────────────────────────
echo "  memoria y anexos..."
for f in "Memoria_TFM_KAN-REC.docx" "Anexos_TFM_KAN-REC.docx"; do
    if [ -f "$RAIZ/docs/$f" ]; then
        cp "$RAIZ/docs/$f" "$SALIDA/memoria/"
    else
        echo "    AVISO: falta docs/$f"
    fi
done
cp "$RAIZ/docs/MEMORIA_TFM_KAN-REC.md" "$SALIDA/memoria/" 2>/dev/null || true
cp "$RAIZ/docs/ANEXOS_TFM_KAN-REC.md"  "$SALIDA/memoria/" 2>/dev/null || true

# ── 2. Codigo fuente ───────────────────────────────────────────────────────
echo "  codigo..."
for d in kanrec tests fabric colab experiments powerbi data infra scripts; do
    [ -d "$RAIZ/$d" ] && cp -r "$RAIZ/$d" "$SALIDA/codigo/"
done
for f in setup.py pytest.ini README.md SECURITY.md .github; do
    [ -e "$RAIZ/$f" ] && cp -r "$RAIZ/$f" "$SALIDA/codigo/"
done
# El .lock documenta las versiones exactas con las que se obtuvieron los resultados
[ -f "$RAIZ/requirements.lock" ] && cp "$RAIZ/requirements.lock" "$SALIDA/codigo/"

# Limpieza: nada de cache, entornos ni datos pesados
find "$SALIDA/codigo" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "$SALIDA/codigo" -name "*.pyc" -delete 2>/dev/null || true
find "$SALIDA/codigo" -name ".venv" -type d -exec rm -rf {} + 2>/dev/null || true
find "$SALIDA/codigo" -name "*.tsv" -delete 2>/dev/null || true
find "$SALIDA/codigo" -name "*.pt" -delete 2>/dev/null || true
find "$SALIDA/codigo" -name ".env" -delete 2>/dev/null || true

# ── 3. Resultados (las cifras que sustentan la memoria) ────────────────────
echo "  resultados..."
for f in "$RAIZ"/docs/*.png; do [ -f "$f" ] && cp "$f" "$SALIDA/resultados/"; done
for f in "$RAIZ"/resultados/*.csv "$RAIZ"/resultados/*.json; do
    [ -f "$f" ] && cp "$f" "$SALIDA/resultados/"
done

# ── 4. Indice del entregable ───────────────────────────────────────────────
cat > "$SALIDA/LEEME.md" << EOF
# $NOMBRE

**Autor:** Pedro Antonio Martínez Sánchez
**Máster:** Big Data & Data Engineering (UCM / NTIC)
**Tutores:** Jorge Centeno y Alberto González

## Repositorio

Código completo, historial de desarrollo e integración continua:
$REPO_URL

## Contenido

    memoria/      Memoria (20 caras) y anexos A-F, en .docx y .md
    codigo/       Paquete kanrec, notebooks de Fabric y Colab, tests
    resultados/   Figuras y CSV con las cifras que sustentan la memoria

## Reproducción

    cd codigo
    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev,train]"
    pytest tests/ -q                    # 105 tests

Los experimentos completos se reproducen con \`experiments/run_all.sh\`.
El orden de ejecución en Fabric y la configuración del pipeline están en
\`codigo/fabric/PIPELINE_README.md\`.

## Nota sobre los datos

El dataset Criteo no se incluye por tamaño (2,26 GB). Se descarga con
\`codigo/data/download_criteo.sh\`.
EOF

# ── 5. Comprobaciones antes de cerrar ──────────────────────────────────────
echo ""
echo "=== Comprobaciones ==="
FALLOS=0

if find "$SALIDA" -name "*.env" -o -name ".env" | grep -q .; then
    echo "  ERROR: hay ficheros .env en el entregable"; FALLOS=1
fi
# Se excluye este propio script: contiene el patron de busqueda y se
# copia dentro del entregable, lo que provocaba un falso positivo.
if grep -rlqs "mongodb+srv://[^<$]" "$SALIDA/codigo" \
       --exclude="empaquetar_entrega.sh" 2>/dev/null; then
    echo "  ERROR: posible credencial de Atlas en el codigo:"
    grep -rln "mongodb+srv://[^<$]" "$SALIDA/codigo" \
         --exclude="empaquetar_entrega.sh" 2>/dev/null | sed 's/^/    /'
    FALLOS=1
fi
if [ ! -f "$SALIDA/memoria/Memoria_TFM_KAN-REC.docx" ]; then
    echo "  ERROR: falta la memoria en .docx"; FALLOS=1
fi
if [ ! -f "$SALIDA/codigo/requirements.lock" ]; then
    echo "  AVISO: falta requirements.lock (ejecuta: pip freeze > requirements.lock)"
fi
if ! ls "$SALIDA/resultados"/*.png >/dev/null 2>&1; then
    echo "  AVISO: no hay figuras en resultados/"
fi

[ "$FALLOS" -eq 1 ] && { echo ""; echo "Corrige los ERROR antes de entregar."; exit 1; }
echo "  sin errores"

# ── 6. Comprimir ───────────────────────────────────────────────────────────
cd "$RAIZ"
zip -r -q "$NOMBRE.zip" entrega -x "*.DS_Store"
TAM=$(du -h "$NOMBRE.zip" | cut -f1)

echo ""
echo "=== Listo: $NOMBRE.zip ($TAM) ==="
unzip -l "$NOMBRE.zip" | tail -1
echo ""
echo "Pendiente aparte del zip: el enlace del video de YouTube (en oculto,"
echo "no en privado) en la tarea del concurso de becas."
