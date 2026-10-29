#!/bin/bash
set -euo pipefail
TASK=/data/x00951564/codex-hy4-20261027-oldhost-revalidation
EXPORT=/mnt/share/x00951564/codex-transfer-20261027/hy4-codex-a5-cann9.1-20261027-verified.tar.gz
TAG=hy4-codex:a5-cann9.1-20261027-verified
test ! -e "$EXPORT"
test ! -e "$EXPORT.partial"
printf 'exporting %s\n' "$TAG" > "$TASK/logs/resume-exclusive/image-export.state"
docker image save "$TAG" | pigz -1 -p 8 > "$EXPORT.partial"
printf 'checking gzip\n' > "$TASK/logs/resume-exclusive/image-export.state"
pigz -t "$EXPORT.partial"
mv "$EXPORT.partial" "$EXPORT"
printf 'computing checksum\n' > "$TASK/logs/resume-exclusive/image-export.state"
sha256sum "$EXPORT" > "$EXPORT.sha256"
cp "$EXPORT.sha256" "$TASK/model_openbox/hy4/IMAGE-ARCHIVE.sha256"
docker image inspect "$TAG" --format '{{.Id}} {{.Size}}' > "$TASK/logs/resume-exclusive/image-export-image-id.txt"
stat -c '%s' "$EXPORT" > "$TASK/logs/resume-exclusive/image-export-bytes.txt"
printf 'passed\n' > "$TASK/logs/resume-exclusive/image-export.state"
printf 'Archive: %s\n' "$EXPORT"
cat "$EXPORT.sha256"
