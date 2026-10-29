set -e
TASK=/data/x00951564/codex-hy4-20261027-oldhost-revalidation
for round in 1 2 3; do
  port=$((29610 + round))
  sed "s/29574/$port/; s/hccl-packed-staged.json/hccl-reinit-$round.json/" "$TASK/scripts/resume-run_hccl_packed_staged.sh" > "$TASK/scripts/resume-hccl-reinit-$round.sh"
  docker exec x00951564-hy4-codex-oldhost-producer bash -c "bash /codex/scripts/resume-hccl-reinit-$round.sh > /codex/logs/resume-exclusive/hccl-reinit-$round-client.log 2>&1"
  echo "HCCL_REINIT_ROUND_$round passed"
done
