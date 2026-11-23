# VGO Cinema - Deployment Guide

## 🚀 Quick Start

### Option A: Docker (recommended)

```bash
docker-compose up --build
```

- Frontend → http://localhost:3000
- Backend  → http://localhost:8081/docs
- Health   → http://localhost:8081/api/health
- Redis    → localhost:6379

### Option B: Local dev (Windows)

```bash
# install backend deps once
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# install frontend deps once
cd ..\frontend
npm install

# start both services
cd ..
python launcher.py
```

Then open:
- Frontend → http://localhost:3000  (auto-bumps to 3001/3002 if busy)
- Backend  → http://localhost:8081/docs
- Logs     → `backend.log` / `frontend.log` in project root

> **Note**: the launcher no longer hard-codes absolute paths; it derives
> paths from `__file__`, so the project can live anywhere.

---

## 🛠️ Architecture Flow

```
User Input → Next.js → FastAPI → Celery (Redis) → [Director → Artist → Animator → Editor] → Final Video
                            ↑                                          ↓
                            └────────── polling  ←────────── task_store.update_task
```

| Stage | Status enum | Where |
|---|---|---|
| 1. Scripting    | `scripting` | `workflow.start_generation_workflow` → `_script_to_text` (NVIDIA `meta/llama-3.3-70b-instruct`) |
| 2. Storyboard   | `image_gen` | `_enhance_prompt` + `image_gen.build_image_url` (NVIDIA `meta/llama-3.1-8b-instruct` + Picsum) |
| 3. Animation    | `video_gen` | motion-description LLM call; clip URL is a stand-in (no free video-gen API) |
| 4. Composition  | `composing` | `workflow.compose_narrative` (NVIDIA `qwen/qwen3.5-397b-a17b`) |
| 5. Done / Fail  | `completed` / `failed` | — |

---

## 📺 How to Test

1. Enter a cinematic prompt on the homepage.
2. Watch the timeline advance `Scripting → Storyboarding → Animating → Composing`.
3. Preview the storyboard images.
4. Watch / download the final video.

---

## ⚙️ Configuration

> ### ⚠️ Before deploying to a shared environment
>
> 1. **Rotate every key pair** that has appeared in chat, screenshots,
>    pull-request descriptions, or `git log -p` output. The bundled
>    `backend/.env` is shipped with **placeholder values only** — fill
>    in your own after rotating.
> 2. The free NVIDIA tier caps you at **40 requests/minute per IP**.
>    The worker pipeline is intentionally serial for image generation
>    so it stays below that ceiling — see `OPTIMIZATION_REPORT.md` if
>    you need to lift the cap (paid plan, multi-IP, or queue-fronted
>    load-balancer).
> 3. If you expose the backend publicly, **always set
>    `CORS_ALLOW_ORIGINS`** to an explicit comma-separated list rather
>    than falling back to the dev regex.

All secrets live in `backend/.env`. A safe template is in `backend/.env.example`.
At minimum set `NVIDIA_API_KEY` to enable real LLM calls (get a free key
at https://build.nvidia.com). The other NVIDIA model defaults are
`meta/llama-3.3-70b-instruct` (director), `meta/llama-3.1-8b-instruct`
(prompt enhancer), and `qwen/qwen3.5-397b-a17b` (composer).

CORS:
- If `CORS_ALLOW_ORIGINS` env is set, it is used verbatim (comma-separated).
- Otherwise the dev regex allows any `http://localhost:{3000,3001,3002,9000,8080}`.

Port map:

| Service  | Container | Host   |
|----------|-----------|--------|
| Frontend | 3000      | 3000   |
| Backend  | 8000      | 8081   |
| Redis    | 6379      | 6379   |
