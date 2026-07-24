#!/bin/bash
# Convenience wrapper, API build: sources env, runs the API pipeline with whatever flags you
# pass through, then scores every result file. All run_typo_api.py flags are accepted.
#
#   bash api-setup/run.sh --dataset gsm8k --variant both --limit 20
#   bash api-setup/run.sh --dataset all --configs all --limit 50 --n-samples 3
#
# Long sweeps survive a closed laptop lid the same way as on GCP: add --detach (or -d) as the
# FIRST argument to relaunch in its own session (setsid+nohup) with a timestamped log.
#
#   bash api-setup/run.sh --detach --dataset math500 --configs all --n-samples 5
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
SELF="$HERE/$(basename "$0")"
OUTDIR="${OUTDIR:-$HOME/nlp_project/results}"
mkdir -p "$OUTDIR"

# --- detach mode: keep running even if the terminal closes -----------------------
if [ "${1:-}" = "--detach" ] || [ "${1:-}" = "-d" ]; then
  shift
  LOGF="$OUTDIR/run_$(date +%Y%m%d_%H%M%S).log"
  setsid nohup bash "$SELF" "$@" > "$LOGF" 2>&1 < /dev/null &
  PID=$!
  echo "detached: PID $PID"
  echo "log:      $LOGF"
  echo "follow:   tail -f $LOGF"
  echo "stop:     kill $PID"
  exit 0
fi

source "$HERE/env.sh"

echo "### inference ###  (started $(date '+%F %T'))"
python3 "$HERE/run_typo_api.py" --outdir "$OUTDIR" "$@"

echo
echo "### scoring ###"
python3 "$HERE/score.py" --results "$OUTDIR"
echo "### done ($(date '+%F %T')) ###"
