#!/usr/bin/env bash
# Clean-stop helper:
#   - kills uvicorn / next processes owned by pids recorded
#     in scripts/logs/*.pid (the up.sh-managed stack),
#   - kills any orphan uvicorn / next from this project (for the
#     case where the stack was started in tmux / detached and the pid
#     files were lost on reboot),
#   - `docker compose down` if the docker daemon is reachable.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"
LOG_DIR="$HERE/logs"

log() { printf '[down.sh] %s\n' "$*"; }

kill_tree() {
    local label="$1" pattern="$2"
    # Pattern is the Win32 image arg substring; we use tasklist to
    # mimic a "ps+grep+kill" idiom that survives the Windows shell.
    local pids
    pids="$(powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe' or Name='node.exe'\" | Where-Object { \$_.CommandLine -like '*${pattern}*' } | Select-Object -ExpandProperty ProcessId" 2>/dev/null | tr -d '\r ' || true)"
    if [[ -n "$pids" ]]; then
        for pid in $pids; do
            log "killing orphan $label pid=$pid"
            taskkill //PID "$pid" //T //F 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
        done
    fi
}

kill_pidfile() {
    local pf="$1"
    if [[ -f "$pf" ]]; then
        local pid
        pid="$(cat "$pf" 2>/dev/null || true)"
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            log "stopping pid=$pid from $pf"
            taskkill //PID "$pid" //T //F 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$pf"
    fi
}

log "managed stack:"
kill_pidfile "$LOG_DIR/backend.pid"
kill_pidfile "$LOG_DIR/frontend.pid"

log "scrubbing orphan Python:"
kill_tree "uvicorn"  "uvicorn backend.main:app"
log "scrubbing orphan node/next:"
kill_tree "next"     "next dev"
kill_tree "node"     "next/dist/server/lib/start-server.js"

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    log "docker stack:"
    (cd "$PROJECT_ROOT" && docker compose down --remove-orphans || true)
fi

log "done."
