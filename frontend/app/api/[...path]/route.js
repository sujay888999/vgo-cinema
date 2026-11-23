// H1: Same-origin API proxy. The browser calls /api/* on this Next.js
// server; the server forwards to the backend with the API key from the
// server-only env (BACKEND_API_KEY). The key never enters the browser bundle.
export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const BACKEND_URL = (process.env.BACKEND_URL || 'http://127.0.0.1:8081').replace(/\/$/, '');
const API_KEY = process.env.BACKEND_API_KEY || '';

const HOP_BY_HOP = new Set([
  'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
  'te', 'trailer', 'transfer-encoding', 'upgrade', 'host',
]);

async function proxy(request, params) {
  const path = (params?.path || []).join('/');
  const incoming = new URL(request.url);
  const target = `${BACKEND_URL}/api/${path}${incoming.search}`;
  const headers = new Headers();
  request.headers.forEach((value, key) => {
    const k = key.toLowerCase();
    if (!HOP_BY_HOP.has(k) && k !== 'content-length') headers.set(key, value);
  });
  if (API_KEY) headers.set('X-Api-Key', API_KEY);
  const method = request.method;
  let body;
  if (method !== 'GET' && method !== 'HEAD') {
    const buf = await request.arrayBuffer();
    if (buf.byteLength > 0) body = buf;
  }
  let upstream;
  try {
    upstream = await fetch(target, { method, headers, body, redirect: 'manual' });
  } catch (e) {
    return Response.json({ detail: 'Backend unavailable' }, { status: 502 });
  }
  const outHeaders = new Headers();
  upstream.headers.forEach((value, key) => {
    if (!HOP_BY_HOP.has(key.toLowerCase())) outHeaders.set(key, value);
  });
  return new Response(upstream.body, { status: upstream.status, headers: outHeaders });
}

export async function GET(request, ctx) { return proxy(request, await ctx.params); }
export async function POST(request, ctx) { return proxy(request, await ctx.params); }
export async function PUT(request, ctx) { return proxy(request, await ctx.params); }
export async function PATCH(request, ctx) { return proxy(request, await ctx.params); }
export async function DELETE(request, ctx) { return proxy(request, await ctx.params); }
export async function HEAD(request, ctx) { return proxy(request, await ctx.params); }
