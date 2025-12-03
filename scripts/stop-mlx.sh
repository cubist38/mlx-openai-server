#!/usr/bin/env bash
# stop-mlx.sh — find and stop mlx-openai-server / hub daemon processes
# Usage: ./scripts/stop-mlx.sh
# NOTE: review the matched commands before confirming.

set -euo pipefail

# Pattern to match likely MLX daemon processes. Adjust if you run under different names.
PATTERN='mlx-openai-server|uvicorn.*app\.hub\.daemon|uvicorn.*app\.main'

echo "Searching for processes matching: $PATTERN"
# List matching processes (PID + full command). Prefer pgrep -af for direct PID+command output.
# `pgrep -af` is available on macOS and Linux and avoids grepping ps output.
matches=$(pgrep -af -- "$PATTERN" 2>/dev/null || true)

if [ -z "$matches" ]; then
  echo "No matching mlx-openai-server/uvicorn processes found."
  exit 0
fi

echo "Found the following processes:"
echo "$matches"
echo

read -r -p "Kill these processes? [y/N]: " confirm
case "$confirm" in
  [Yy]) ;;
  *)
    echo "Aborting — no processes were killed."
    exit 0
    ;;
esac

# Try graceful shutdown first
echo "Sending SIGTERM to processes matching: $PATTERN"
pkill -TERM -f "$PATTERN" 2>/dev/null || echo "Failed to SIGTERM some processes"

# Wait up to TIMEOUT seconds for processes to exit
TIMEOUT=10
echo "Waiting up to $TIMEOUT seconds for processes to exit..."
count=$TIMEOUT
while [ $count -gt 0 ]; do
  remaining=$(pgrep -af -- "$PATTERN" 2>/dev/null || true)
  if [ -z "$remaining" ]; then
    echo "All processes exited cleanly."
    exit 0
  fi
  sleep 1
  count=$((count - 1))
done

# Escalate to SIGKILL for any remaining
echo "Timed out waiting for graceful exit; sending SIGKILL to remaining processes"
pkill -KILL -f "$PATTERN" 2>/dev/null || echo "Failed to KILL some processes"

echo "Done. Re-run the script or use 'pgrep -af \"$PATTERN\"' to verify no matching processes remain."
