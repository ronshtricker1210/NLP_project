#!/bin/bash
# Convenience wrapper for the GCP L4 instance: sources env, runs the vLLM pipeline with whatever
# flags you pass through, then scores every result file. All run_typo_vllm.py flags are accepted.
#
#   bash gcp-setup/run.sh --dataset gsm8k --variant both --limit 20
#   bash gcp-setup/run.sh --dataset all --configs all --limit 50 --n-samples 3
#
# Survive SSH disconnect: add --detach (or -d) as the FIRST argument. The job is relaunched in
# its own session (setsid+nohup), detached from the terminal, and keeps running after you log out.
# Output goes to a timestamped log; the command prints the PID, log path, and how to follow it.
#
#   bash gcp-setup/run.sh --detach --dataset gsm8k --variant both --limit 50
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
SELF="$HERE/$(basename "$0")"
OUTDIR="${OUTDIR:-$HOME/nlp_project/results}"
mkdir -p "$OUTDIR"

# --- detach mode: keep running even if the SSH connection drops ----------------
if [ "${1:-}" = "--detach" ] || [ "${1:-}" = "-d" ]; then
  shift
  LOGF="$OUTDIR/run_$(date +%Y%m%d_%H%M%S).log"
  # setsid: new session (no controlling terminal) · nohup: ignore SIGHUP · </dev/null: detach stdin
  setsid nohup bash "$SELF" "$@" > "$LOGF" 2>&1 < /dev/null &
  PID=$!
  echo "detached: PID $PID   (keeps running after SSH disconnect / logout)"
  echo "log:      $LOGF"
  echo "follow:   tail -f $LOGF | tr '\\r' '\\n'"
  echo "stop:     kill $PID"
  exit 0
fi

source "$HERE/env.sh"

echo "### inference ###  (started $(date '+%F %T'))"
python3 "$HERE/run_typo_vllm.py" --outdir "$OUTDIR" "$@"

echo
echo "### scoring ###"
python3 "$HERE/score.py" --results "$OUTDIR"
echo "### done ($(date '+%F %T')) ###"
