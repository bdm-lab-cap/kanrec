#!/bin/bash
# Downloads Criteo Display Advertising dataset and creates a 10M-row subsample.
# Requires: Kaggle CLI configured  (pip install kaggle + ~/.kaggle/kaggle.json)
set -e

echo "=== Downloading Criteo dataset ==="
kaggle competitions download -c criteo-display-ad-challenge -f train.txt.gz -p data/
gunzip -f data/train.txt.gz

MD5_EXPECTED="b22b63027a2dd5c6e1af75f5a5d07aad"
MD5_ACTUAL=$(md5sum data/train.txt | cut -d' ' -f1)
if [ "$MD5_EXPECTED" != "$MD5_ACTUAL" ]; then
    echo "ERROR: MD5 mismatch. File may be corrupt."
    exit 1
fi
echo "MD5 OK: $(du -sh data/train.txt)"

echo "Creating 10M-row subsample..."
head -n 10000001 data/train.txt > data/criteo_10m.tsv
echo "Subsample ready: data/criteo_10m.tsv ($(du -sh data/criteo_10m.tsv | cut -f1))"
