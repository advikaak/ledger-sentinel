#!/usr/bin/env bash
# Full pipeline, start to finish. About two and a half minutes.
set -e
cd "$(dirname "$0")"
mkdir -p data models out/figures

echo "[1/7] generating the vendor ledger"
python3 src/generate_data.py

echo
echo "[2/7] training"
python3 src/train.py

echo
echo "[3/7] evaluating"
python3 src/evaluate.py

echo
echo "[4/7] writing the website's data (docs/data/site-data.js)"
python3 src/export_site.py

echo
echo "[5/7] executing the analysis notebook (figures -> out/figures/)"
(cd notebooks && python3 -m nbconvert --to notebook --execute --inplace analysis.ipynb --ExecutePreprocessor.record_timing=False --log-level=WARN)

echo
echo "[6/7] tests"
python3 -m pytest -q

echo
echo "[7/7] agent demo"
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "    ANTHROPIC_API_KEY not set -- running without Claude narration"
  python3 src/demo.py --no-llm
else
  python3 src/demo.py
fi
