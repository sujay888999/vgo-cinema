# =============================================================================
# VGO Cinema - Backend (multi-stage)
#
# Stage 1 (builder) : installs build deps (gcc, headers) and pinned wheels.
# Stage 2 (runtime) : slim image with only the runtime libraries needed to
#                     import the project site-packages.
# The lockfile (`requirements.lock`) is generated with `pip freeze` and
# pinned to the exact versions tested in development.
# =============================================================================

# ---- Stage 1: builder -------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pinned deps into a fixed prefix we can copy from in stage 2.
COPY requirements.lock /srv/requirements.lock
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r /srv/requirements.lock

# ---- Stage 2: runtime -------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/install/bin:/usr/local/bin \
    PYTHONPATH=/install/lib/python3.11/site-packages

WORKDIR /srv

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash app

# Bring across the site-packages from the builder.
COPY --from=builder /install /install
# Bring across the project source.
COPY --chown=app:app . /srv/backend

USER app
WORKDIR /srv

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
