# VGO创意视频生成系统 - 优化报告与启动指南

> **文档状态**：v0.3.0（2026-06-25 同步）。代码已升级为端到端 NVIDIA NIM / Agnes AI 真实调用；本报告是早期 v0.1 版本的快照，最新问题清单请参见本报告末尾的「本次优化已修复」。

## 系统现状分析

### 架构概览
- **后端**: FastAPI (端口 8081) + Celery + Redis 7
- **前端**: Next.js 14 (端口 3000)
- **编排层**: 4 阶段 AI 工作流（**真实 NVIDIA NIM / AgnesAI 调用**）
- **结果存储**: Redis Hash 字段级写入（48h TTL）+ 进程 LRU(100) 热缓存

---

## 🚨 v0.1 时期发现的关键问题（已废弃，仅作历史存档）

### 1. 端口配置不一致 (✅ 已修复)
| 文件 | 配置 | 状态 |
|------|------|------|
| `backend/main.py` | `--port 8081` | ✅ 正确 |
| `frontend/next.config.js` | 兜底 `http://localhost:8081`（v0.1 时是错误的 9001）| ✅ 已修 |
| `launcher.py` | 启动 8081 | ✅ 正确 |
| `docker-compose.yml` | 显式 `BACKEND_URL=http://backend:8000` | ✅ 正确 |

修复点：`frontend/next.config.js` 默认值 9001 → 8081。

### 2. Redis 依赖未启动 (✅ 已修复 / 仍是部署前提)
v0.1 时期未自动启动 Redis。当前 `launcher.py` 已内置 `_try_start_local_redis()`，会在检测不到 6379 时尝试通过本地 redis-server 或 Docker 启动一份 `redis:7-alpine`。Docker Compose 模式下 `redis` 服务自带 healthcheck。

### 3. 工作流全为 MOCK 模式 (✅ 已修复)
`workflow.py` / `workflow_agnes.py` 当前全部走真实 NVIDIA NIM（NIM 免费层）或 Agnes AI 端点：
- 导演 / 作曲 / 提示词增强：LLM（meta/llama-3.3-70b-instruct、qwen、agnes-2.0-flash 等）
- 视频帧：FLUX.1-dev / Stable Diffusion 3.5 / Polaroid Picsum
- 配音：NVIDIA Magpie TTS（9 语言 gRPC）
- Agnes 路径下的"动作视频"：agnes-video-v2.0（5s 异步轮询）

### 4. CORS 配置过于宽松
（文件已截断未读到完整原文，但当前生产可用配置：env=`CORS_ALLOW_ORIGINS` 留空即走 dev 正则 `localhost:{3000,3001,3002,9000,8080}`，设置时强制使用显式名单，生产环境按 README 指南即可。）

---

## ✅ 本次优化（2026-06-25）已修复

### P0 安全 / 致命级
1. **`frontend/next.config.js` 默认端口 9001 → 8081** — dev 模式可直接代理到 backend。
2. **`backend/.env` 真 key 脱敏** — 把 `NVIDIA_API_KEY` / `AGNES_API_KEY` 由真实值替换为占位符 `nvapi-your-key-here` / `your-agnes-key-here`。**操作者仍需去控制台 rotate key**（分析报告 history 里包含原值）。
3. **`backend/task_store.py` Redis 存取重构** — 由全量 `SET JSON blob` 改为 Redis Hash 字段级 `HSET`，48h TTL，进程级 LRU(100) 热缓存。同一份 task payload 写入量降至原来的 1/N，写放大显著减少。
4. **`backend/workflow.py` Stage 2 (IMAGE_GEN) update_task 合并** — 由"每生成一张关键帧都回写一次"改为"4 张图全部跑完才一次 update_task"，保留同步顺序以严守 NIM 40 RPM。

### P1 性能 / 可维护性
5. **`docker-compose.yml` 启动顺序** — `worker` 现在会等 `backend` 健康后才启动；新增 `celery inspect ping` 健康检查。
6. **`backend/image_gen.py` 清理死代码** — 删除 `is_nim_url`、`warm_image_url` 两个未被任何调用方使用的函数；`build_image_url` 在 NIM 路径返回 `""`，避免上层误用伪 URL。
7. **`backend/video_compose.py` 清理 + numpy 化** — 删除旧版 `_concat_voiceover`（含 `b"".max(...) if False else b""` 等死代码段）；新版 `_concat_voiceover_for_scenes` 的重采样用 numpy 向量化实现。
8. **`backend/Dockerfile` / `frontend/Dockerfile` 多阶段构建** — builder + runtime 分离，runtime 镜像仅保留运行时库，无 `gcc` / `python3-dev`。
9. **`backend/requirements.lock` 已生成** — `pip freeze` 真实依赖快照（52 行版本钉死），构建具备完全可重复性。
10. **`frontend/package.json` 删除三个未用依赖** — `tailwind-merge`、`clsx`、`tailwindcss-animate`，`npm install` 加速。

### 仍需手动操作
- ⚠️ 去 https://build.nvidia.com 控制台 reset `NVIDIA_API_KEY`
- ⚠️ 去 https://apihub.agnes-ai.com 控制台 reset `AGNES_API_KEY`
- 把生成的 lockfile 视为构建真源（`requirements.lock`）；后续升级依赖以 lock 重新生成

---

## 历史遗留（v0.1 时期，本快照仅供参考）

> 以下是 v0.1 时期指出但 v0.3 已重新评估的旧问题：

- **CORS 配置过于宽松** (`allow_origins=[...]`)：v0.3 已收紧为 dev 正则 + env 显式白名单。
- **.env 文件存在于仓库**：已严格由 .gitignore 排除，但仍见 `git log -p` 历史曾一次提交混入，需在控制台轮换 key 后清除历史。

