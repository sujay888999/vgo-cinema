# =============================================================================
# VGO Cinema - Frontend (multi-stage)
#
# Stage 1 (deps)    : install npm deps (cached as a layer).
# Stage 2 (build)   : build the Next.js application.
# Stage 3 (runtime) : tiny production-only node runtime.
# =============================================================================

# ---- Stage 1: deps ----------------------------------------------------------
FROM node:18-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund

# ---- Stage 2: build ---------------------------------------------------------
FROM node:18-alpine AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

ARG NEXT_PUBLIC_API_BASE=http://localhost:8081/api
ENV NEXT_PUBLIC_API_BASE=${NEXT_PUBLIC_API_BASE}

RUN npm run build

# ---- Stage 3: runtime -------------------------------------------------------
FROM node:18-alpine AS runtime
WORKDIR /app
ENV NODE_ENV=production

# Only carry what `next start` needs at runtime.
COPY --from=build /app/public ./public
COPY --from=build /app/.next ./.next
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/package.json ./package.json
COPY --from=build /app/next.config.js ./next.config.js

EXPOSE 3000
CMD ["npm", "start"]
