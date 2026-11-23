#!/usr/bin/env bash
# VGO Cinema — cold-start launcher.
#
# Behaviour:
#   1. If `docker compose` is reachable AND the compose stack can boot a
#      healthy backend (with the in-process worker), we prefer that.
#   2. Otherwise (Docker daemon down / not installed / Linux engine broken),
#      we bring up host processes:
#        - WSL Ubuntu redis if not already running, on its eth0 IP.
#        - uv run uvicorn backend.main:app on http://127.0.0.1:8081
#           — the FastAPI app now owns an internal ThreadPoolExecutor
#             that runs workflows in-process. No separate Celery
#             worker is required.
#        - npm run dev in frontend/   (port 3000)
#   3. Polls /api/health up to 30 s and exits 0 only when the API is live.
#
# Idempotent — running twice will refuse to start a second stack while the
# PID files exist. Use `down.sh` first if you actually want to restart.
#
# Usage:
#   ./scripts/up.sh          # detached (logs in scripts/logs/)
#   ./scripts/up.sh --fg     # keep tailing logs in foreground
#   ./scripts/up.sh --down   # drop a running stack and exit
#
# Idempotent — running twice will refuse to start a second stack while the
# PID files exist. Use `down.sh` first if you actually want to restart.
#
# Usage:
#   ./scripts/up.sh          # detached (logs in scripts/logs/)
#   ./scripts/up.sh --fg     # keep tailing logs in foreground
#   ./scripts/up.sh --down   # drop a running stack and exit
set -euo pipefail

# ---------------------------------------------------------------- paths -----
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"
LOG_DIR="$HERE/logs"
mkdir -p "$LOG_DIR"
PID_BACKEND="$LOG_DIR/backend.pid"
PID_FRONTEND="$LOG_DIR/frontend.pid"
LOG_BACKEND="$LOG_DIR/backend.log"
LOG_FRONTEND="$LOG_DIR/frontend.log"

MODE="detached"
case "${1:-}" in
    --fg)      MODE="foreground" ;;
    --down)    MODE="down" ;;
    "")        MODE="detached" ;;
    *)         echo "unknown flag: $1" >&2; exit 64 ;;
esac

# ---------------------------------------------------------------- helpers ----
log() { printf '[up.sh] %s\n' "$*"; }

is_pid_alive() {
    local pid="${1:-}"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

kill_pidfile() {
    local pf="$1"
    if [[ -f "$pf" ]]; then
        local pid
        pid="$(cat "$pf" 2>/dev/null || true)"
        if is_pid_alive "$pid"; then
            log "stopping pid=$pid from $pf"
            # Tree-kill so child uvicorn sub-processes go too.
            taskkill //PID "$pid" //T //F 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$pf"
    fi
}

docker_compose_reachable() {
    command -v docker >/dev/null 2>&1 || return 1
    # `docker info` exits non-zero if the daemon is dead even if the
    # CLI is on PATH. We accept any docker context that can produce a
    # successful `info`.
    docker info >/dev/null 2>&1
}

# ---------------------------------------------------------------- down --------
if [[ "$MODE" == "down" ]]; then
    log "--down: tearing down host stack"
    kill_pidfile "$PID_BACKEND"
    kill_pidfile "$PID_FRONTEND"
    log "done."
    exit 0
fi

# ---------------------------------------------------------------- pre-check ---
# Refuse to start a second backend if the PID file is alive.
if is_pid_alive "$(cat "$PID_BACKEND" 2>/dev/null)"; then
    log "stack seems already up; aborting. Use --down first."
    exit 1
fi

cd "$PROJECT_ROOT"

# ---------------------------------------------------------------- docker -----
if docker_compose_reachable; then
    log "docker daemon reachable; using docker-compose"
    docker compose up -d backend frontend || {
        log "docker compose failed; falling through to host stack" >&2
        # fall through on error
        true
    }
    if [[ $? -eq 0 ]]; then
        # Wait for backend healthcheck via HTTP
        for i in $(seq 1 30); do
            if curl -s -m 2 http://127.0.0.1:8081/api/health >/dev/null 2>&1; then
                log "stack ready (compose mode)"
                exit 0
            fi
            sleep 1
        done
        log "compose stack didn't become healthy in 30 s. Run: docker compose logs"
        exit 1
    fi
fi

# ---------------------------------------------------------------- ws-redis ---
log "docker unreachable; bringing up WSL redis + host stack"
WSL_REDIS_IP="$(
    wsl -d Ubuntu bash -c 'echo $REMOTE_IP' 2>/dev/null || true
)"
# Default: rely on cached value from .env; otherwise probe.
REDIS_URL="${REDIS_URL:-$(grep '^REDIS_URL=' backend/.env 2>/dev/null | cut -d= -f2-)}"
if [[ -z "$REDIS_URL" || "$REDIS_URL" == "redis://localhost"* ]]; then
    # Try probing WSL for an eth0 IP and start redis there if needed.
    WSL_IP="$(wsl -d Ubuntu bash -c "hostname -I" 2>/dev/null | awk '{print $1}' || true)"
    if [[ -z "$WSL_IP" ]]; then
        log "WSL Ubuntu missing or empty hostname -I. Aborting." >&2
        exit 2
    fi
    REDIS_URL="redis://${WSL_IP}:6379/0"
    # Make sure redis-server is running and bound to 0.0.0.0 inside WSL.
    wsl -d Ubuntu bash -c '
        redis-cli shutdown nosave 2>/dev/null || true
        sleep 0.3
        cat > /tmp/redis.conf <<CFG
bind 0.0.0.0
protected-mode no
port 6379
daemonize yes
save ""
pidfile /tmp/redis.pid
logfile /tmp/redis.log
loglevel notice
CFG
        redis-server /tmp/redis.conf || apt-get install -y redis-server >/dev/null && redis-server /tmp/redis.conf
        sleep 0.5
    '
    # Probe TCP from Windows host.
    if ! powershell -NoProfile -Command "Test-NetConnection -ComputerName $WSL_IP -Port 6379 -InformationLevel Quiet -WarningAction SilentlyContinue 2>\$null" >/dev/null; then
        log "redis at $WSL_IP:6379 unreachable from Windows host. Aborting." >&2
        exit 3
    fi
fi
export REDIS_URL
log "REDIS_URL=$REDIS_URL"

# ---------------------------------------------------------------- backend ----
log "starting backend (uvicorn) → $LOG_BACKEND"
(
    cd "$PROJECT_ROOT"
    PYTHONPATH=. ./backend/venv/Scripts/python.exe -m uvicorn backend.main:app \
        --host 0.0.0.0 --port 8081 --log-level info \
        >"$LOG_BACKEND" 2>&1 &
    echo $! > "$PID_BACKEND"
)

# The workflow now runs inside `backend.main`'s ThreadPoolExecutor — a
# separate Celery worker is unnecessary, so we skip the celery launch
# here entirely. See backend/main.py `local_pool`.

# ---------------------------------------------------------------- frontend --
log "starting next.js frontend → $LOG_FRONTEND"
(
    cd "$PROJECT_ROOT/frontend"
    # Use `npx --yes` so node is allowed to download deps if missing.
    npx --yes next dev -p 3000 >"$LOG_FRONTEND" 2>&1 &
    echo $! > "$PID_FRONTEND"
)

# ---------------------------------------------------------------- health -----
log "polling /api/health for up to 30 s ..."
ready=0
for i in $(seq 1 30); do
    if curl -s -m 2 http://127.0.0.1:8081/api/health >/dev/null 2>&1; then
        log "API up after ${i} s"
        ready=1
        break
    fi
    sleep 1
done
if [[ "$ready" -ne 1 ]]; then
    log "ERROR: API did not become healthy."
    log "  tail backend :  $LOG_BACKEND"
    exit 4
fi

# ---------------------------------------------------------------- mode ------
[[ "$MODE" == "detached" ]] && {
    log "stack ready (host mode). Logs: $LOG_DIR"
    log "/api/health OK ; frontend http://127.0.0.1:3000 ; backend http://127.0.0.1:8081"
    exit 0
}
log "foreground mode: tailing logs (Ctrl+C to break out, stack keeps running)"
trap 'echo; log "Foreground exited; stack still up. Use --down to stop."; exit 0' INT TERM
tail -n +1 -F "$LOG_BACKEND" "$LOG_FRONTEND"
