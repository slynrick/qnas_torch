#!/bin/bash
# Runs `make queue-watch` on a remote host over ssh and reconnects automatically
# when the connection drops (ssh exits 255 on connection loss/timeouts), with a
# growing delay between attempts. Quitting the dashboard (Ctrl+C) or any
# non-connection failure ends it for good.
#
# Usage: scripts/watch-remote.sh <ssh_host> <remote_project_path> [watch args...]
#   scripts/watch-remote.sh puc_dualgpu1 /home/fcardoso/projects/qnas_torch --interval 2

set -u

if [ $# -lt 2 ]; then
    echo "usage: $0 <ssh_host> <remote_project_path> [watch args...]" >&2
    exit 2
fi
host=$1
remote_path=$2
shift 2
watch_args="$*"

delay=2
max_delay=30
attempt=0
while true; do
    ssh -t -C -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o ConnectTimeout=10 \
        "$host" "cd '${remote_path}' && make queue-watch WATCH_ARGS='${watch_args}'"
    rc=$?
    if [ "$rc" -ne 255 ]; then
        exit "$rc"
    fi
    attempt=$((attempt + 1))
    printf '\n[%s] connection to %s lost (attempt %d) - retrying in %ds (Ctrl+C to quit)...\n' \
        "$(date +%H:%M:%S)" "$host" "$attempt" "$delay" >&2
    sleep "$delay" || exit 130
    delay=$((delay * 2 > max_delay ? max_delay : delay * 2))
done
