#!/usr/bin/env bash
# Live view of a pipeline run (scripts/run_pipeline.sh) in the terminal; refreshes every INTERVAL seconds, Ctrl-C quits.
#
#   scripts/watch.sh                                  # latest dataset, backend from progress logs (cuda first)
#   scripts/watch.sh data/datasets/ds-... cuda        # explicit
#   INTERVAL=10 LINES=25 scripts/watch.sh
#
# Shows: whether the runner is alive, GPU use, the pipeline's step history (progress.log), and the last lines of the
# step that is running now (training: update/total, loss, ETA; scoring: objects/total, ETA).
set -uo pipefail
cd "$(dirname "$0")/.."
DS=${1:-$(ls -d data/datasets/ds-* 2>/dev/null | tail -1)}
[ -n "$DS" ] || { echo "no dataset in data/datasets/ (run make stage-a first)"; exit 1; }
NAME=$(basename "$DS")
B=${2:-}
if [ -z "$B" ]; then
  for b in cuda mlx cpu; do [ -f "reports/pipeline/$NAME/$b/progress.log" ] && { B=$b; break; }; done
fi
LOG=reports/pipeline/$NAME/${B:-cuda}
INTERVAL=${INTERVAL:-5}
LINES=${LINES:-20}

while true; do
  out=$(
    echo "=== $NAME  backend=${B:-?}  $(date '+%F %T')  (refresh ${INTERVAL}s, Ctrl-C to quit) ==="
    if pgrep -f "run_pipeline.sh" >/dev/null; then echo "runner: RUNNING"; else echo "runner: not running"; fi
    if command -v nvidia-smi >/dev/null; then
      nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu \
                 --format=csv,noheader | sed 's/^/GPU:    /'
    fi
    echo
    if [ -f "$LOG/progress.log" ]; then
      echo "--- steps ($LOG/progress.log) ---"
      tail -n 15 "$LOG/progress.log"
      cur=$(awk '$3=="START"{s=$4} $3=="DONE"&&$4==s{s=""} $3=="FAILED"{s=""} END{print s}' "$LOG/progress.log")
      last_failed=$(grep " FAILED " "$LOG/progress.log" | tail -1)
      echo
      if [ -n "$cur" ] && [ -f "$LOG/$cur.log" ]; then
        echo "--- running: $cur ($LOG/$cur.log) ---"
        tr '\r' '\n' < "$LOG/$cur.log" | grep -v '^\s*$' | tail -n "$LINES"
      elif [ -n "$last_failed" ] && tail -1 "$LOG/progress.log" | grep -q FAILED; then
        f=$(echo "$last_failed" | awk '{print $4}')
        echo "--- FAILED: $f — last lines of $LOG/$f.log ---"
        tr '\r' '\n' < "$LOG/$f.log" 2>/dev/null | grep -v '^\s*$' | tail -n "$LINES"
      fi
    else
      echo "no $LOG/progress.log yet — start it with: make pipeline BACKEND=cuda DS=$DS"
    fi
  )
  clear
  echo "$out"
  sleep "$INTERVAL"
done
