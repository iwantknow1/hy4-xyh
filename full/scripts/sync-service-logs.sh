#!/bin/bash
set -euo pipefail
node=${1:?node number}
runtime=${2:?runtime directory}
run=${3:-04}
end=$((SECONDS+28800))
out=/task/logs/node$node
for ((attempt=0; attempt<60; attempt++)); do
  if [[ -f "$runtime/_serve.sh" && -f "$runtime/stdout.log" && -f "$runtime/stderr.log" ]]; then
    break
  fi
  sleep 2
done
cp "$runtime/_serve.sh" "$out/dp4-command-$run.sh"
printf '%s\n' "$runtime" > "$out/dp4-runtime-$run.txt"
while (( SECONDS < end )); do
  for stream in stdout stderr; do
    cp "$runtime/$stream.log" "$out/full-dp4-$run.$stream.log.tmp"
    mv "$out/full-dp4-$run.$stream.log.tmp" "$out/full-dp4-$run.$stream.log"
  done
  sleep 30
done
