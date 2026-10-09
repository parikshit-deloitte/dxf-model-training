#!/usr/bin/env bash
# Stages B -> C -> D -> F -> registry -> E for one dataset, one step at a time, on any backend:
#
#   scripts/run_pipeline.sh data/datasets/ds-YYYYMMDD-xxxxxxxx                 # BACKEND=auto (mlx on a Mac, cuda on NVIDIA)
#   BACKEND=cuda SEEDS="20261007 7" scripts/run_pipeline.sh data/datasets/ds-...
#
# Order: B1/B2 baselines (CPU; skipped when their outputs exist) -> for the first seed: train, calibrate, package,
# register (usable in trial mode from here) -> zero-shot calibration (what the gate compares against; cached scores
# resume) -> gate + promotion attempt -> B3 zero-shot LOPO baseline (informational) -> the other seeds.
# One step at a time: they share one GPU. Logs and progress.log: reports/pipeline/<ds>/<backend>/. A step whose
# DONE line is in progress.log is skipped, so the script can be re-run after a failure or a stop.
# The registry promotes only on a PASS gate run against the current active model.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} PYTHONPATH=src
PY=${PY:-.venv/bin/python}
DS=${1:?usage: [BACKEND=auto|mlx|cuda] [SEEDS="20261007 7"] scripts/run_pipeline.sh data/datasets/ds-...}
NAME=$(basename "$DS")
B=$(DXFTRAIN_BACKEND=${BACKEND:-} $PY -m dxftrain.infer.backend)        # resolved backend: mlx | cuda | cpu
export DXFTRAIN_BACKEND=$B
read -r -a SEED_LIST <<< "${SEEDS:-20261007 7}"
LOG=reports/pipeline/$NAME/$B
mkdir -p "$LOG"
ZRUN=$($PY -m dxftrain.infer.backend --zero-shot-dir "$DS")
echo "$(date '+%F %T') PIPELINE backend=$B seeds=${SEED_LIST[*]} host=$(hostname)" >> "$LOG/progress.log"

done_already() { grep -q " DONE $1\$" "$LOG/progress.log" 2>/dev/null; }
step() {
  local name=$1; shift
  if done_already "$name"; then echo "skip $name (done)"; return 0; fi
  echo "$(date '+%F %T') START $name" >> "$LOG/progress.log"
  if "$@" > "$LOG/$name.log" 2>&1; then
    echo "$(date '+%F %T') DONE $name" >> "$LOG/progress.log"
  else
    echo "$(date '+%F %T') FAILED $name (see $LOG/$name.log)" >> "$LOG/progress.log"; exit 1
  fi
}
latest_run() { ls -d runs/"$B"-*-s"$1" 2>/dev/null | while read -r r; do
  grep -q "\"dataset\": \"$NAME\"" "$r/run.json" && grep -q '"best_epoch": [0-9]' "$r/run.json" && echo "$r"; done | tail -1; }
train_seed() {
  if [ "$B" = mlx ]; then $PY -u -m dxftrain.train.mlx_train --dataset "$DS" --seed "$1"
  else $PY -u -m dxftrain.train.train_lora --dataset "$DS" --seed "$1"; fi
}
served_arg() {  # what the gate's G3 compares: MLX serves a fused artefact; CUDA serves the evaluated base + adapter
  if [ "$B" = mlx ]; then echo "models/$(basename "$1")/served"; else echo same; fi
}
model_path() {  # seed -> trained, calibrated, packaged and registered (usable in trial mode)
  local seed=$1 run
  step "train_s$seed" train_seed "$seed"
  run=$(latest_run "$seed")
  [ -n "$run" ] || { echo "$(date '+%F %T') FAILED no finished $B run for seed $seed" >> "$LOG/progress.log"; exit 1; }
  echo "$(date '+%F %T') RUN s$seed $run" >> "$LOG/progress.log"
  step "stage_d_s$seed" $PY -u -m dxftrain.calibrate.stage_d --dataset "$DS" --run "$run"
  step "package_s$seed" $PY -u -m dxftrain.package.package --run "$run"
  if [ "$B" = mlx ]; then
    step "register_s$seed" $PY -m dxftrain.registry.registry register --run "$run" --served "models/$(basename "$run")/served"
  else
    step "register_s$seed" $PY -m dxftrain.registry.registry register --run "$run"
  fi
}
gate_path() {  # seed -> Stage E gate against the baselines, then promotion (refused unless PASS)
  local seed=$1 run
  run=$(latest_run "$seed")
  step "gate_s$seed" $PY -u -m dxftrain.gate.gate --dataset "$DS" --run "$run" --zero-shot-run "$ZRUN" \
       --served "$(served_arg "$run")"
  if $PY -m dxftrain.registry.registry promote "$(basename "$run")" >> "$LOG/promote_s$seed.log" 2>&1; then
    echo "$(date '+%F %T') PROMOTED $(basename "$run")" >> "$LOG/progress.log"
  else
    echo "$(date '+%F %T') NOT PROMOTED $(basename "$run") (gate not PASS; see promote_s$seed.log)" >> "$LOG/progress.log"
  fi
}

SB=reports/stage_b/$NAME
if [ -f "$SB/B1_rules.no_colour.preds.jsonl" ] && [ -f "$SB/B2_gbm.with_colour.preds.jsonl" ]; then
  echo "skip stage_b_b1b2 (outputs exist in $SB)"
else
  step stage_b_b1b2 $PY -u -m dxftrain.baselines.stage_b --dataset "$DS" --only b1,b2
fi
FIRST=${SEED_LIST[0]}
model_path "$FIRST"                                  # usable in trial mode from here (MODEL_ID=<run id> make serve)
step stage_d_zeroshot $PY -u -m dxftrain.calibrate.stage_d --dataset "$DS" --zero-shot --backend "$B"
gate_path "$FIRST"
step stage_b_b3 $PY -u -m dxftrain.baselines.stage_b --dataset "$DS" --only b3 --backend "$B"
for seed in "${SEED_LIST[@]:1}"; do
  model_path "$seed"
  gate_path "$seed"
done
echo "$(date '+%F %T') ALL DONE" >> "$LOG/progress.log"
