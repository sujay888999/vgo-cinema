# VGO Cinema - Google Colab 部署脚本
# ==============================================================================
# 使用方法：
#   1. 在 Colab (https://colab.research.google.com/) 创建新 Notebook
#   2. 上传 vgo-cinema.zip 并解压到 /vgo-cinema
#   3. 依次运行以下单元格
#
# 注意：
#   - Colab 免费版 VM 约 12 小时后断开，需要重新连接
#   - 无持久化存储，重启后需重新部署
#   - 默认使用 mock 模式（无需 API 密钥），实际生成视频需配置密钥
# ==============================================================================

import os, sys, subprocess, time, pathlib

PROJECT_DIR = "/vgo-cinema"
BACKEND_PORT = 8081
FRONTEND_PORT = 3000

# ==============================================================================
# 单元格 1：安装系统依赖
# ==============================================================================
print("[1/5] 安装系统依赖...")
!apt-get update -qq
!apt-get install -y -qq ffmpeg git curl nodejs npm
print("✅ 系统依赖安装完成")

# ==============================================================================
# 单元格 2：安装 Python 依赖
# ==============================================================================
print("[2/5] 安装 Python 依赖...")
!pip install -r {PROJECT_DIR}/backend/requirements.lock
print("✅ Python 依赖安装完成")

# ==============================================================================
# 单元格 3：配置环境
# ==============================================================================
print("[3/5] 配置环境...")

# 创建输出目录
!mkdir -p /vgo-outputs

# 写入 .env 配置（mock 模式，无需 API 密钥）
env_content = """# VGO Cinema - Colab 部署配置
ENABLE_REDIS=false
AUTH_REQUIRED=false
API_KEYS=
CORS_ALLOW_ORIGINS=*
LOG_LEVEL=INFO
MAX_SCENES_PER_VIDEO=4
VIDEO_MODE=auto
VGO_OUTPUT_DIR=/vgo-outputs

# === 如需实际生成视频，请填写以下密钥 ===
# NVIDIA NIM (https://build.nvidia.com)
# NVIDIA_API_KEY=

# Agnes AI
# AGNES_API_KEY=

# 其他可选配置
# PROVIDER=agnes
# VIDEO_PROVIDER_PRIMARY=agnes
# VIDEO_PROVIDER_FALLBACK=mock,null
"""
pathlib.Path(f"{PROJECT_DIR}/backend/.env").write_text(env_content)
print("✅ 环境配置完成")

# ==============================================================================
# 单元格 4：启动后端
# ==============================================================================
print("[4/5] 启动后端服务...")

# 终止可能已存在的后端进程
!pkill -f "uvicorn backend.main:app" 2>/dev/null || true
time.sleep(1)

# 后台启动 uvicorn
!cd {PROJECT_DIR}/backend && nohup python -m uvicorn backend.main:app --host 0.0.0.0 --port {BACKEND_PORT} > /vgo-backend.log 2>&1 &

# 等待后端启动
import urllib.request
backend_ready = False
for i in range(30):
    try:
        resp = urllib.request.urlopen(f"http://localhost:{BACKEND_PORT}/api/health", timeout=2)
        if resp.status == 200:
            backend_ready = True
            print(f"✅ 后端已启动！({i+1}次尝试)")
            break
    except:
        time.sleep(2)

if not backend_ready:
    print("❌ 后端启动失败，查看日志：")
    print(!cat /vgo-backend.log)
else:
    # 验证 API
    try:
        resp = urllib.request.urlopen(f"http://localhost:{BACKEND_PORT}/api/health", timeout=3)
        print(f"健康检查: HTTP {resp.status}")
        print(f"响应: {resp.read().decode()}")
    except Exception as e:
        print(f"健康检查异常: {e}")

# ==============================================================================
# 单元格 5：启动前端
# ==============================================================================
print("[5/5] 启动前端服务...")

# 终止可能已存在的前端进程
!pkill -f "next start" 2>/dev/null || true
time.sleep(1)

# 安装前端依赖
print("安装前端依赖...")
!cd {PROJECT_DIR}/frontend && npm install --no-audit --no-fund

# 构建前端
print("构建前端...")
!cd {PROJECT_DIR}/frontend && NEXT_PUBLIC_API_BASE=http://localhost:{BACKEND_PORT}/api npm run build

# 后台启动前端
!cd {PROJECT_DIR}/frontend && NEXT_PUBLIC_API_BASE=http://localhost:{BACKEND_PORT}/api nohup npm start -- -p {FRONTEND_PORT} > /vgo-frontend.log 2>&1 &

# 等待前端启动
for i in range(30):
    try:
        resp = urllib.request.urlopen(f"http://localhost:{FRONTEND_PORT}", timeout=2)
        if resp.status == 200:
            print(f"✅ 前端已启动！({i+1}次尝试)")
            break
    except:
        time.sleep(2)

# ==============================================================================
# 部署完成
# ==============================================================================
print("\n" + "="*50)
print("  ✅ VGO Cinema 部署完成！")
print("="*50)
print(f"后端 API:  http://localhost:{BACKEND_PORT}/api")
print(f"前端页面:  http://localhost:{FRONTEND_PORT}")
print(f"后端日志:  !cat /vgo-backend.log")
print(f"前端日志:  !cat /vgo-frontend.log")
print("\n通过 Colab 端口转发访问：")
print(f"  后端: https://[你的Colab地址]/proxy/{BACKEND_PORT}")
print(f"  前端: https://[你的Colab地址]/proxy/{FRONTEND_PORT}")
print("\n停止服务：")
print("  !pkill -f uvicorn; !pkill -f next")
