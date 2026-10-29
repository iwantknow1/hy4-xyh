#!/bin/bash
set -euo pipefail
: "${HALLUSION_JUDGE_API_KEY:?Configure judge credentials before scoring; inference outputs are reusable}"
: "${HY4_HALLUSION_RESULT:?Path to the new model HallusionBench_result.json}"
: "${HY4_JUDGE_OUTPUT:?Use a new judge output directory}"
[[ ! -e "$HY4_JUDGE_OUTPUT" ]] || { echo 'Use a new judge output directory to avoid cached historical scores' >&2; exit 1; }
export HALLUSION_JUDGE_BASE_URL=${HALLUSION_JUDGE_BASE_URL:-https://api.deepseek.com}
export HALLUSION_JUDGE_MODEL=${HALLUSION_JUDGE_MODEL:-deepseek-flash}
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/task/evaluation/benchmark:/mnt/share/z00949185/hy4-work/22.1/aisbench-venv/lib/python3.12/site-packages
mkdir -p "$HY4_JUDGE_OUTPUT"
cp /task/evaluation/hallusion/evaluation.py /task/evaluation/hallusion/utils.py "$HY4_JUDGE_OUTPUT/"
cp "$HY4_HALLUSION_RESULT" "$HY4_JUDGE_OUTPUT/HallusionBench_result.json"
cd "$HY4_JUDGE_OUTPUT"
exec /task/evaluation/.venv-client/bin/python -u evaluation.py
