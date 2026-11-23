# VGO Cinema — AI Video Generation Operating System

> Turn a text prompt into a cinematic short by orchestrating NVIDIA NIM
> LLMs and a free image service. Real models end-to-end (no mocks).

---

## ✨ What's real vs. mocked

| Stage | Provider | Model | Real? |
|---|---|---|---|
| 🎬 **Scripting** (Director) | NVIDIA NIM | `meta/llama-3.3-70b-instruct` | ✅ Real LLM call |
| 🖼️ **Image prompt enhancer** (Artist) | NVIDIA NIM | `meta/llama-3.1-8b-instruct` | ✅ Real LLM call |
| 🎞️ **Animation** (Animator) | NVIDIA NIM | `meta/llama-3.1-8b-instruct` | ⚠️ Ken Burns fallback — NVIDIA free NIM has no video-gen model |
| ✂️ **Composing** (Editor) | NVIDIA NIM | `meta/llama-3.3-70b-instruct` | ✅ Real LLM call |
| 🖼️ **Image rendering** | FLUX / Picsum / your provider | `flux_dev` (default) | ✅ FLUX via NVIDIA NIM is prompt-matched; Picsum is a fallback |

> See **Plug in real image / video generation** below to wire up Stable Diffusion, DALL·E, Runway, Luma, etc.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Next.js 14  (Frontend, Tailwind + Zustand + TanStack Query)         │
│  http://localhost:3000                                                │
└──────────────────────────┬───────────────────────────────────────────┘
                           │ HTTP
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  FastAPI  (Pydantic v2 + pydantic-settings)                           │
│  http://localhost:8081/docs                                            │
│   ├─ POST /api/generate   → enqueue                                  │
│   ├─ GET  /api/tasks/{id}                                             │
│   ├─ GET  /api/tasks                                                  │
│   └─ GET  /api/health                                                 │
└──────────────────────────┬───────────────────────────────────────────┘
                           │ local_pool.submit()
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Redis 7  (task store: Hash + LRU cache, 48h TTL)                     │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  In-process ThreadPoolExecutor — workflow.start_generation_workflow  │
│                                                                       │
│   Stage 1  meta/llama-3.3-70b-instruct  →  script (JSON)             │
│   Stage 2  meta/llama-3.1-8b-instruct   →  image prompts            │
│             image_gen.build_image_url()  →  picsum/SD URL           │
│   Stage 3  meta/llama-3.1-8b-instruct   →  motion descriptions      │
│   Stage 4  qwen/qwen3.5-397b-a17b       →  final narrative          │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Option A — Docker (recommended)

```bash
docker-compose up --build
```

| Service  | URL                          |
|----------|------------------------------|
| Frontend | http://localhost:3000        |
| Backend  | http://localhost:8081/docs   |
| Health   | http://localhost:8081/api/health |
| Redis    | localhost:6379               |

A valid `NVIDIA_API_KEY` must be available as an env var. In `docker-compose.yml`
the variable is read from the host environment, so:

```bash
export NVIDIA_API_KEY=nvapi-...     # Linux / macOS
setx NVIDIA_API_KEY nvapi-...       # Windows (then restart shell)
docker-compose up --build
```

### Option B — Local development (Windows)

```bash
# 1. backend deps
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# 2. frontend deps
cd ..\frontend
npm install

# 3. start a Redis instance
docker run -d -p 6379:6379 redis:7-alpine

# 4. start both services
cd ..
python launcher.py
```

The launcher spawns two new console windows and writes logs to
`backend.log` and `frontend.log` in the project root.
Task execution runs in-process via a ThreadPoolExecutor (no separate
Celery worker).

Then open:
- Frontend → http://localhost:3000  (auto-bumps to 3001/3002 if busy)
- Backend  → http://localhost:8081/docs
- Logs     → `backend.log` / `frontend.log`

---

## ⚡ Cold-start / stop launcher (recommended)

Two cross-platform shell scripts live under `scripts/` and work from
MSYS bash, git-bash, WSL, or any native `bash` on the host. They detect
whether Docker is reachable:

| Mode | Trigger | Behaviour |
|---|---|---|
| **Docker** | `docker info` succeeds | `docker compose up -d backend frontend`; PID/health checks forwarded to compose |
| **Host fallback** | Docker daemon down / missing | Bootstraps Redis inside WSL Ubuntu on its eth0 IP, then runs `uvicorn` + `next dev` on the host |

```bash
# 1) start everything (polls /api/health for up to 30 s before exiting 0)
./scripts/up.sh

# 2) follow all three logs in your terminal
./scripts/up.sh --fg

# 3) stop everything (pids in scripts/logs/*.pid + orphan uvicorn/celery/next)
./scripts/down.sh
```

Logs:
- `scripts/logs/backend.log` — uvicorn stdout/stderr
- `scripts/logs/frontend.log` — next dev

PID files (`backend.pid`, `frontend.pid`) are written so
you can `kill --pid <pid>` directly from the shell or from CI jobs.

The launcher is idempotent — running `up.sh` again without first running
`down.sh` is a no-op (refuses to double-bind the same backend port).

---

## ⚙️ Configuration

> ### ⚠️ Security checklist (read first!)
>
> 1. **Never commit `backend/.env`** — it ships with `<placeholder>` values by
>    default; replace them with keys you obtain directly from the providers.
> 2. **Rotate any key that has appeared in chat, screenshots, or `git log -p`
>    history.** The free NVIDIA tier is rate-limited to **40 requests/minute
>    per IP**; if you notice 429 responses during a heavy run, switch to a
>    paid plan or add an upstream queue (the pipeline is intentionally
>    serial for image-gen to stay under that cap — see
>    `OPTIMIZATION_REPORT.md`).
> 3. **Source-restrict the key at the provider portal** where possible
>    (NVIDIA NIM allows IP allow-listing on paid plans).

## ⚡ Live progress: SSE + auth + rate limiting

Once the stack is up you can drive the system from the React UI (chat-style)
or from `curl`:

```bash
# Optional: send an API key to get a per-key rate-limit bucket.
# When unset, the IP becomes the bucket.
X_API_KEY="sk-test:demo:60"

# Submit a generation
curl -X POST http://127.0.0.1:8081/api/generate \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: $X_API_KEY" \
  -d '{"prompt":"A small wooden boat on misty lake at sunrise, painterly, slow cinematic","style":"cinematic"}'
# → 202 Accepted
# {"task_id":"<uuid>","status":"pending","style":"cinematic"}

# Stream live progress
curl -N http://127.0.0.1:8081/api/tasks/<uuid>/events | head -5
# data: {"type":"ready", "task_id":"...", ...}
# data: {"type":"snapshot", "status":"image_gen", "scenes_state":[...], "eta_seconds":124.0, ...}
# data: {"type":"update", "status":"video_gen", ...}

# Inspect a saved task
curl http://127.0.0.1:8081/api/tasks/<uuid>

# Who am I and how many requests do I have left?
curl -H "X-Api-Key: $X_API_KEY" http://127.0.0.1:8081/api/me
```

**Authentication** is opt-in. Set `AUTH_REQUIRED=true` to require an
`X-Api-Key` header; leave it off for a local dev loop. The key bucket
is keyed per-token; otherwise IP becomes the bucket.

```
Backing variable ........ Where you set ............... Default
───────────────────── ───────────────────────── ─────────────────
PROVIDER               .env / docker-compose       "agnes"
AGNES_API_KEY          .env / docker-compose       "" (required for Agnes)
NVIDIA_API_KEY         .env / docker-compose       "" (required for NVIDIA)
AUTH_REQUIRED          .env                       false
API_KEYS               .env                       ""
                       key:label:limit_per_window
                       "sk-prod:demo:10,sk-test:local:60"
RATE_LIMIT_DEFAULT     .env                       10
RATE_LIMIT_WINDOW_SEC  .env                       60
```

**SSE events carry enough state for the UI to animate per-scene chips
without polling:**

```
data: {"type":"snapshot", "status":"video_gen",
       "scenes_state":[{"scene":1,"image_ready":true,"video_ready":true,"audio_ready":false},
                       {"scene":2,"image_ready":true,"video_ready":false,"audio_ready":false},
                       {...}],
       "scenes_count":4, "scenes_completed":2,
       "eta_seconds":145.2, "stage_durations_sec":{...},
       "ts": 1782390000.0}
```

The SSE stream is **not silent-safe**: if the Redis broker flaps on a
flaky connection (e.g. WSL2 NAT on a Windows host), the endpoint will
keep emitting frames by polling the task store directly. So you'll
never see the UI hang on a `pending` chip — it will animate forward.

---

## ⚡ Live progress: SSE + auth + rate limiting — quick reference

| Key | Default | Notes |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379/0` | Task store (Redis Hash + LRU) |
| `CORS_ALLOW_ORIGINS` | _(empty)_ | Empty → dev regex `localhost:{3000,3001,3002,9000,8080}` |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `MAX_SCENES_PER_VIDEO` | `4` | Capped at 8 |
| `NVIDIA_API_KEY` | _(empty)_ | **Required** to enable real LLM calls. Get one free at https://build.nvidia.com |
| `NVIDIA_DIRECTOR_MODEL` | `meta/llama-3.3-70b-instruct` | Strongest 70B in the free tier |
| `NVIDIA_COMPOSER_MODEL` | `meta/llama-3.3-70b-instruct` | Fast + reliable on free NIM tier |
| `NVIDIA_PROMPT_MODEL` | `meta/llama-3.1-8b-instruct` | Small/fast prompt enhancer |
| `IMAGE_PROVIDER` | `flux_dev` | `flux_dev` / `flux_schnell` / `sd35` / `picsum` / `pollinations` / `none` |

> ⚠️ All three NVIDIA models above are non-reasoning chat models — reasoning
> variants (Nemotron-*, etc.) put the answer in `reasoning_content` rather
> than `content` and would not work with the current `nvidia_client.py`.

---

## 🔌 Plug in real image / video generation

NVIDIA's free NIM tier does not currently include any text-to-image or
text-to-video model. The pipeline therefore uses **Picsum** (real photos,
prompt-independent) for the storyboard so the UI looks alive, and the
animation stage records a motion description but does not produce a real
clip. To upgrade:

### Image generation

Edit `backend/image_gen.py` and add a new provider:

```python
def _build_stable_diffusion_url(prompt: str, seed: str, s) -> str:
    return (
        f"https://your-sd-host/generate?prompt={urllib.parse.quote(prompt)}"
        f"&seed={seed}&width={s.image_width}&height={s.image_height}"
    )
```

Then in `build_image_url()` add an `elif provider == "stable-diffusion"` branch
and set `IMAGE_PROVIDER=stable-diffusion` in `.env`.

### Video generation

Add a Runway / Luma / Kling call in `backend/workflow.py` inside
`Stage 3 (VIDEO_GEN)`. The current code uses Ken Burns fallback for
video; replace the `scene.video_url = img_url` assignment with a real
async submission and poll for the clip URL. The code already stores
motion descriptions per scene so the new provider can consume them.

---

## 📁 Project Layout

```
.
├── backend/                # FastAPI + ThreadPoolExecutor + NVIDIA NIM
│   ├── main.py             # HTTP API
│   ├── config.py           # pydantic-settings (env, models, image provider)
│   ├── nvidia_client.py    # OpenAI-compatible wrapper for NVIDIA NIM
│   ├── image_gen.py        # Picsum / Pollinations / custom providers
│   ├── workflow.py         # 4-stage cinematic pipeline (real LLMs)
│   ├── task_store.py       # Redis Hash task store + LRU cache
│   ├── schemas.py          # Pydantic models
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
│
├── frontend/               # Next.js 14 (App Router)
│   ├── app/
│   │   ├── page.jsx        # Cinematic Generator UI
│   │   ├── layout.js       # Root layout + QueryClient provider
│   │   ├── providers.jsx   # TanStack Query client
│   │   └── globals.css
│   ├── store/useVideoStore.ts   # Zustand store
│   ├── package.json
│   └── Dockerfile
│
├── docker-compose.yml      # 3-service stack
├── launcher.py             # Windows quick start
├── start.bat               # Wraps launcher.py
├── DEPLOYMENT.md           # Detailed deploy guide
└── README.md               # This file
```

---

## 🧪 Status

> **v0.3.0** — the pipeline is now driven by real NVIDIA NIM LLMs end-to-end.
>
> ✅ Director LLM produces structured cinematic scripts (JSON)
> ✅ Image-prompt enhancer LLM produces Flux-ready prompts
> ✅ Composer LLM produces the final narrative (markdown)
> ✅ Task state machine (scripting → image_gen → video_gen → composing → completed/failed)
> ✅ Frontend polls real progress with TanStack Query
> ✅ Docker stack is reproducible
> ✅ Graceful degradation: if a stage fails, the task is marked FAILED and
>    a descriptive error is returned; the workflow retries up to 2 times.
>
> 🔜 Real image generation (Stable Diffusion / DALL·E / FLUX)
> 🔜 Real video generation (Runway / Luma / Kling)
> ✅ Redis Hash task store with LRU cache (48h TTL)
> ✅ SSE for real-time progress updates
> 🔜 Auth + rate limiting + asset storage
