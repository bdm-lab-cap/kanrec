#!/bin/bash
# Higiene del repositorio: elimina los restos que el informe externo detecto.
# Ejecutar desde la raiz del repo. Revisar la salida antes de hacer commit.
set -e
echo "=== Ficheros basura de 'pip install pkg>=x' sin comillas ==="
for f in '=1.12.0' '=1.4.0' '=2.2.0' '=15.0.0' '=4.6.0' '=2.11.0'; do
    [ -e "$f" ] && { echo "  eliminando '$f'"; git rm -f --cached "$f" 2>/dev/null || true; rm -f "$f"; }
done

echo "=== Copias con sufijo ==="
for f in 'README (1).md' 'README (2).md'; do
    [ -e "$f" ] && { echo "  eliminando '$f'"; git rm -f --cached "$f" 2>/dev/null || true; rm -f "$f"; }
done

echo "=== Duplicados en la raiz (la version buena esta en su carpeta) ==="
for f in config.py ci.yml gitignore env env.example 03_api_ingest.py \
         09_mongodb_vector_search.py kafka_producer_confluent.py \
         mongodb_import.py test_config.py test_no_secrets.py; do
    [ -f "$f" ] && { echo "  eliminando ./$f"; git rm -f --cached "$f" 2>/dev/null || true; rm -f "$f"; }
done

echo "=== Directorios de fases anteriores ==="
for d in fabric/fabric databricks build dist; do
    [ -d "$d" ] && { echo "  eliminando $d/"; git rm -rf --cached "$d" 2>/dev/null || true; rm -rf "$d"; }
done
find . -name "*.egg-info" -maxdepth 2 -type d -exec rm -rf {} + 2>/dev/null || true

echo "=== Actualizando .gitignore ==="
for linea in "build/" "dist/" "*.egg-info/" "=*" "*.whl"; do
    grep -qxF "$linea" .gitignore 2>/dev/null || echo "$linea" >> .gitignore
done

echo ""
echo "=== Comprobacion final ==="
echo "Ficheros en la raiz:"
ls -p | grep -v / | sed 's/^/  /'
echo ""
echo "Version del paquete:"
grep -h "__version__\|version=" kanrec/__init__.py setup.py | sed 's/^/  /'
echo ""
echo "Revisa 'git status' antes de hacer commit."
