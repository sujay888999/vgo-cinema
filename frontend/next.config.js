/** @type {import('next').NextConfig} */
// H1: No /api rewrite here on purpose. The App Router route handler at
// app/api/[...path]/route.js is the single /api proxy: it forwards to the
// backend and injects the API key server-side. A next.config.js rewrite
// would forward WITHOUT the key (it shadowed the route handler and caused
// 401s), so it must not exist alongside the route handler.
const nextConfig = {
  reactStrictMode: true,
};

module.exports = nextConfig;
