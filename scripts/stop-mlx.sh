#!/usr/bin/env bash
# stop-mlx.sh — find and stop mlx-openai-server / hub daemon processes
# Usage: ./scripts/stop-mlx.sh
# NOTE: review the matched commands before confirming.

set -euo pipefail

# Pattern to match likely MLX daemon processes. Adjust if you run under different names.
PATTERN='mlx-openai-server|uvicorn.*app\.hub\.daemon|uvicorn.*app\.main'

echo "Searching for processes matching: $PATTERN"
# List matching processes (PID + full command). Use ps to be portable across macOS/Linux.
matches=$(ps -eo pid=,command= | grep -E "$PATTERN" || true)

if [ -z "$matches" ]; then
  echo "No matching mlx-openai-server/uvicorn processes found."
  exit 0
fi

echo "Found the following processes:"
echo "$matches"
echo

# Collect PIDs (space-separated)
pids=$(echo "$matches" | awk '{print $1}' | tr '\n' ' ' | sed 's/ $//')
if [ -z "$pids" ]; then
  echo "No PIDs found. Exiting."
  exit 0
fi

echo "PIDs: $pids"
read -r -p "Kill these processes? [y/N]: " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
  echo "Aborting — no processes were killed."
  exit 0
fi

# Try graceful shutdown first
echo "Sending SIGTERM to: $pids"
for pid in $pids; do
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || echo "Failed to SIGTERM $pid"
  else
    echo "Process $pid not running"
  fi
done

# Wait up to TIMEOUT seconds for processes to exit
TIMEOUT=10
echo "Waiting up to $TIMEOUT seconds for processes to exit..."
count=$TIMEOUT
while [ $count -gt 0 ]; do
  alive=false
  for pid in $pids; do
    if kill -0 "$pid" 2>/dev/null; then
      alive=true
      break
    fi
  done
  if [ "$alive" = false ]; then
    echo "All processes exited cleanly."
    exit 0
  fi
  sleep 1
  count=$((count - 1))
done

# Escalate to SIGKILL for any remaining
echo "Timed out waiting for graceful exit; sending SIGKILL to remaining PIDs"
for pid in $pids; do
  if kill -0 "$pid" 2>/dev/null; then
    echo "KILL $pid"
    kill -KILL "$pid" 2>/dev/null || echo "Failed to KILL $pid"
  fi
done

echo "Done. Re-run the script or use 'pgrep -af \"$PATTERN\"' to verify no matching processes remain."
