# === Stage 1: Build Frontend ===
FROM oven/bun:1 AS builder

WORKDIR /build

# Copy workspace config and source files
COPY package.json bun.lock CHANGELOG.md ./
COPY app/ ./app/
COPY web/ ./web/

# Strip unused workspace configurations and clean package.json
RUN sed -i '/"tauri"/d; /"landing"/d' package.json && \
    sed -i -z 's/,\n  ]/\n  ]/' package.json

# Install workspace dependencies
RUN bun install --no-save

# Build the web app (skip tsc compile check because of upstream type errors)
RUN cd web && bunx --bun vite build

# === Stage 2: Serve with Nginx ===
FROM nginx:alpine

# Copy custom nginx config
COPY web/nginx.conf /etc/nginx/conf.d/default.conf

# Copy static assets from build stage
COPY --from=builder /build/web/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
