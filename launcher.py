import os
import subprocess
import sys
import time
from pathlib import Path

# =============================================================================
# VGO CINEMA - PROFESSIONAL SYSTEM LAUNCHER (v2)
# =============================================================================

# Use the directory of this script so the launcher is portable.
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
FRONTEND_DIR = ROOT_DIR / "frontend"

# Auto-detect venv: prefer backend/venv, fall back to project-root venv.
VENV_CANDIDATES = [
    BACKEND_DIR / "venv",
    ROOT_DIR / "venv",
]


def get_python_exe() -> str:
    for path in VENV_CANDIDATES:
        exe = path / "Scripts" / "python.exe"
        if exe.exists():
            return str(exe)
    return sys.executable


def get_npm_exe() -> str:
    """Locate npm.exe (Windows) or 'npm' on PATH."""
    candidates = [
        Path(os.environ.get("APPDATA", "")) / "npm" / "npm.cmd",
        Path("C:/Program Files/nodejs/npm.cmd"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return "npm"


def run_in_new_console(title: str, cwd: Path, cmd_args: list[str], log_name: str) -> None:
    """Spawn a long-running process in a new console window with logging."""
    log_path = ROOT_DIR / f"{log_name}.log"
    log_path.unlink(missing_ok=True)

    if sys.platform == "win32":
        # Use `start` so the spawned process gets its own console.
        cmd_str = " ".join(f'"{a}"' if " " in str(a) else str(a) for a in cmd_args)
        full_cmd = (
            f'start "{title}" cmd /k "cd /d \"{cwd}\" && {cmd_str} > \"{log_path}\" 2>&1"'
        )
        subprocess.Popen(full_cmd, shell=True)
    else:
        log_fp = open(log_path, "ab")
        subprocess.Popen(cmd_args, cwd=cwd, stdout=log_fp, stderr=subprocess.STDOUT)


def _is_redis_up() -> bool:
    """Best-effort: probe localhost:6379 to see if Redis is reachable."""
    try:
        import socket
        with socket.create_connection(("127.0.0.1", 6379), timeout=0.5):
            return True
    except OSError:
        return False


def _redis_password() -> str:
    """Read REDIS_PASSWORD from backend/.env (H2). Empty if unset."""
    import os
    try:
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend", ".env")
        for line in open(env_path):
            if line.startswith("REDIS_PASSWORD="):
                return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return ""


def _try_start_local_redis() -> None:
    """Try to spawn a local redis-server / docker redis so the worker has a broker.
    Best-effort: if neither is available we just print a warning."""
    import shutil
    redis_pw = _redis_password()
    if not redis_pw:
        print("  [redis] WARNING: REDIS_PASSWORD is not set in backend/.env; starting Redis without a password")
    if True:  # H2: try local redis-server on any platform (was win32-only)
        redis_exe = shutil.which("redis-server")
        if redis_exe:
            subprocess.Popen(
                [redis_exe, "--port", "6379", "--bind", "127.0.0.1"] + (["--requirepass", redis_pw] if redis_pw else []) + ["--save", "", "--appendonly", "no"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            print("  [redis] spawning local redis-server on 127.0.0.1:6379 (password-protected)")
            return
        docker = shutil.which("docker")
        if docker:
            subprocess.Popen(
                [docker, "run", "--rm", "-d", "--name", "vgo-redis",
                 "redis:7-alpine", "redis-server", "--bind", "127.0.0.1", "--appendonly", "no"] + (["--requirepass", redis_pw] if redis_pw else []),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            print("  [redis] spawning redis:7-alpine in docker (loopback only, password-protected)")
            return
    print("  [redis] no local redis found and no docker — please start Redis manually")
    print("      e.g. redis-server --port 6379 --bind 127.0.0.1 --requirepass <REDIS_PASSWORD from backend/.env>")


def launch() -> None:
    print("=" * 70)
    print("       VGO CINEMA - SYSTEM BOOTSTRAP  (v2)")
    print("=" * 70)
    print(f"Project root : {ROOT_DIR}")
    print(f"Backend dir  : {BACKEND_DIR}")
    print(f"Frontend dir : {FRONTEND_DIR}")
    print("=" * 70)

    if not BACKEND_DIR.exists():
        print(f"[ERROR] Backend directory not found: {BACKEND_DIR}")
        sys.exit(1)
    if not FRONTEND_DIR.exists():
        print(f"[ERROR] Frontend directory not found: {FRONTEND_DIR}")
        sys.exit(1)

    python_exe = get_python_exe()
    print(f"Using Python : {python_exe}")
    print(f"Using npm    : {get_npm_exe()}")

    if not _is_redis_up():
        print("\n[!] Redis not detected on 127.0.0.1:6379 — attempting to start one...")
        _try_start_local_redis()
        time.sleep(2)
        if not _is_redis_up():
            print("[!] Redis still unreachable; backend will crash-loop until you start it.")

    # 1/2 — FastAPI backend (task execution is in-process via ThreadPoolExecutor).
    backend_args = [
        python_exe, "-m", "uvicorn", "backend.main:app",
        "--host", "0.0.0.0", "--port", "8081", "--reload",
    ]
    print("\n[1/2] Launching AI Director Engine (Backend @ :8081)...")
    run_in_new_console("VGO Backend", ROOT_DIR, backend_args, "backend")

    # 2/2 — Next.js dev server.
    npm_exe = get_npm_exe()
    frontend_args = [npm_exe, "run", "dev"]
    print("[2/2] Launching Cinematic Interface (Frontend @ :3000+)...")
    run_in_new_console("VGO Frontend", FRONTEND_DIR, frontend_args, "frontend")

    print("\n" + "=" * 70)
    print("  ALL SYSTEMS ONLINE!")
    print("  FRONTEND : http://localhost:3000  (auto-bump to 3001/3002 if busy)")
    print("  BACKEND  : http://localhost:8081/docs")
    print("  HEALTH   : http://localhost:8081/api/health")
    print("=" * 70)
    print("  Logs: backend.log / frontend.log in project root.")
    print("  Open the new console windows for real-time output.")


if __name__ == "__main__":
    try:
        launch()
    except Exception as exc:
        print(f"Fatal Error: {exc}")
    time.sleep(5)
