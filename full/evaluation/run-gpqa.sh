#!/usr/bin/env bash
set -euo pipefail
: "${HY4_GPQA_INPUT:?Set the verified exact-prompts JSONL path}"
: "${HY4_MODEL_PATH:?Set the actual full model path}"
: "${HY4_API_HOST:?Set the inference API host}"
: "${HY4_API_PORT:?Set the inference API port}"
: "${HY4_EVAL_OUTPUT:?Set a new output directory}"
: "${HY4_AISBENCH_BIN:?Set the ais_bench entrypoint in the uv-managed evaluation venv}"
[[ -x "$HY4_AISBENCH_BIN" ]] || { echo 'AISBench entrypoint unavailable' >&2; exit 1; }
[[ ! -e "$HY4_EVAL_OUTPUT" ]] || { echo 'Output directory already exists; choose a new run directory' >&2; exit 1; }
actual_hash=$(sha256sum "$HY4_GPQA_INPUT")
[[ "${actual_hash%% *}" == 1705137667c5b9b4e2ad5b3a85a0d94103bad3c303f41aa24609203771a91e2d ]] || { echo 'GPQA input hash mismatch' >&2; exit 1; }
config_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export HY4_GPQA_INPUT HY4_MODEL_PATH HY4_API_HOST HY4_API_PORT HY4_EVAL_OUTPUT
export NO_PROXY='*' no_proxy='*' TOKENIZERS_PARALLELISM=false
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
"$HY4_AISBENCH_BIN" "$config_dir/gpqa-config.py" --mode all \
  --num-prompts 198 --num-warmups 0 --max-num-workers 1 \
  --dump-eval-details --dump-extract-rate --debug
