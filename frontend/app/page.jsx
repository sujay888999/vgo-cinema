"use client";

import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useVideoStore } from '../store/useVideoStore';
import {
  Sparkles, Film, Image as ImageIcon, Loader2, CheckCircle2, AlertCircle, AlertTriangle,
  Camera, Sun, Clock, Hash, FileText, ChevronRight, Aperture, Clapperboard,
  Mic, Music4, Coins, Volume2, Quote,
} from 'lucide-react';

const STAGES = [
  { id: 'scripting', label: 'Scripting',     icon: Clapperboard },
  { id: 'image_gen', label: 'Storyboarding', icon: ImageIcon },
  { id: 'video_gen', label: 'Animating',     icon: Film },
  { id: 'composing', label: 'Composing',     icon: CheckCircle2 },
];
const STAGE_ORDER = ['scripting', 'image_gen', 'video_gen', 'composing', 'completed'];
const TERMINAL = new Set(['completed', 'failed']);

async function fetchTask(apiBase, taskId, apiKey = '') {
  const headers = apiKey ? { 'X-Api-Key': apiKey } : {};
  const res = await fetch(`${apiBase}/tasks/${taskId}`, { headers });
  if (!res.ok) throw new Error(`Task fetch failed: HTTP ${res.status}`);
  return res.json();
}

// Subscribe to live task updates via Server-Sent Events. Returns an
// unsubscribe function. The handler is called on each `snapshot` (initial)
// or `update` event; data is a partial VideoTask-shaped object.
function subscribeTaskEvents(apiBase, taskId, handler, onError, apiKey = '') {
  if (!taskId) return () => {};
  // apiBase already ends in `/api`, so the event stream URL is
  // `${apiBase}/tasks/${taskId}/events`. EventSource cannot set custom
  // headers, so the API key is passed as a ?key= query parameter.
  const url = apiKey
    ? `${apiBase}/tasks/${taskId}/events?key=${encodeURIComponent(apiKey)}`
    : `${apiBase}/tasks/${taskId}/events`;
  if (typeof window === 'undefined' || !window.EventSource) {
    return () => {};
  }
  let closed = false;
  let backoffMs = 1000;
  let current = null;
  let timer = null;
  let retries = 0;
  const MAX_RETRIES = 10; // cap at ~5 minutes of retrying

  const open = () => {
    if (closed || retries > MAX_RETRIES) {
      if (onError && retries > MAX_RETRIES) {
        onError('SSE connection lost after maximum retries.');
      }
      return;
    }
    current = new EventSource(url);
    current.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        handler(data);
        // Reset backoff and retry counter after a healthy frame.
        backoffMs = 1000;
        retries = 0;
      } catch (err) {
        // Ignore malformed frames.
      }
    };
    current.onerror = () => {
      // Browser auto-reconnects in the background, but we still cap
      // retries to busy-loop-with-hope situations. Replace with a
      // reconnect that backs off so we don't hammer the server.
      try { current.close(); } catch (err) {}
      current = null;
      if (closed) return;
      retries++;
      timer = setTimeout(() => {
        backoffMs = Math.min(backoffMs * 2, 8000);
        open();
      }, backoffMs);
    };
  };
  open();
  return () => {
    closed = true;
    if (timer) { clearTimeout(timer); timer = null; }
    if (current) { try { current.close(); } catch (err) {} }
  };
}

function absoluteUrl(apiBase, path) {
  if (!path) return path;
  if (/^https?:\/\//.test(path)) return path;
  const base = (apiBase || '').replace(/\/api\/?$/, '');
  return `${base}${path.startsWith('/') ? path : '/' + path}`;
}

// Validate that a media URL is safe to use in <video src>, <img src>, or
// <a href>. Only allows backend-generated output paths (/api/outputs/*)
// and absolute HTTP(S) URLs to trusted domains. Blocks data: URLs,
// javascript: URLs, and untrusted external domains that could be used
// for XSS or open-redirect attacks.
const TRUSTED_HOSTS = ['127.0.0.1', 'localhost'];
function isSafeMediaUrl(url) {
  if (!url || typeof url !== 'string') return false;
  // Backend-generated output paths are always safe.
  if (url.startsWith('/api/outputs/')) return true;
  // Absolute URLs must be HTTP(S) to a trusted host.
  try {
    const u = new URL(url);
    if (u.protocol !== 'http:' && u.protocol !== 'https:') return false;
    return TRUSTED_HOSTS.includes(u.hostname);
  } catch {
    return false;
  }
}

function fmtTime(sec) {
  if (!Number.isFinite(sec) || sec <= 0) return '0:00';
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

function Frame({ index, scene, status, sceneImageReady, sceneVideoReady, provenance }) {
  // sceneImageReady / sceneVideoReady are *booleans* that the parent
  // tracks per-scene via SSE (it doesn't need the full URL to render the
  // slate — a boolean signals to the chip that this scene is past image /
  // video gen respectively). `provenance` is 'video' | 'kenburns' | 'missing'
  // and is only meaningful once the scene is past VIDEO_GEN (i.e.
  // sceneVideoReady is true). It defaults to undefined until then.
  const SCENE_LABELS = ['idle', 'image', 'video', 'voice', 'mux'];
  const filled = sceneImageReady || !!scene?.image_url;
  const videoReady = sceneVideoReady || !!scene?.video_url;
  const tickIdx = filled ? (videoReady ? 4 : 2) : 0;
  const isFallback = provenance === 'kenburns';
  const isMissing = provenance === 'missing';
  return (
    <div className="group relative flex-shrink-0 w-[280px] bg-amber-50 text-zinc-900 rounded-sm shadow-xl border border-amber-200/60 overflow-hidden">
      {/* Frame header (like a production slate) */}
      <div className="flex items-center justify-between px-3 py-1.5 bg-zinc-900 text-amber-50 font-mono text-[10px] tracking-widest">
        <span className="flex items-center gap-1">
          <Hash className="w-3 h-3" /> {String(index + 1).padStart(2, '0')}
        </span>
        <span className="uppercase opacity-70">
          {scene?.shot_type?.split(' ')[0] || '—'}
        </span>
        <span className="flex items-center gap-1 tabular-nums">
          <Clock className="w-3 h-3" /> {fmtTime(scene?.duration_sec)}
        </span>
      </div>

      {/* Keyframe image */}
      <div className="relative aspect-video bg-amber-100/50">
        {filled ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={scene?.image_url}
            alt={`Scene ${index + 1}`}
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-amber-700/50 gap-2">
            <Aperture className="w-8 h-8 animate-pulse" />
            <span className="text-[10px] font-mono uppercase tracking-widest">
              {status === 'scripting' ? 'pending director' : 'rendering keyframe…'}
            </span>
          </div>
        )}
        {/* Lighting badge overlaid */}
        {scene?.lighting && filled && (
          <div className="absolute top-2 right-2 px-2 py-0.5 bg-black/70 text-amber-100 rounded text-[9px] font-mono uppercase tracking-wider backdrop-blur">
            <Sun className="w-2.5 h-2.5 inline mr-1" />
            {scene.lighting}
          </div>
        )}
        {/* Per-scene-progress chips (lower-left) */}
        <div className="absolute bottom-2 left-2 flex items-center gap-1">
          {SCENE_LABELS.map((lbl, i) => {
            const reached = i <= tickIdx;
            return (
              <span
                key={lbl}
                title={`${lbl}${reached ? ' (done)' : ' (pending)'}`}
                className={`text-[8px] font-mono uppercase px-1.5 py-0.5 rounded-sm tracking-wider transition ${
                  reached
                    ? 'bg-emerald-500/85 text-emerald-950'
                    : i === tickIdx + 1
                      ? 'bg-amber-200/85 text-amber-900 animate-pulse'
                      : 'bg-zinc-900/55 text-zinc-400'
                }`}
              >
                {lbl}
              </span>
            );
          })}
        </div>
      </div>

      {/* Caption / action lines */}
      <div className="px-3 py-2.5 space-y-1.5 border-t border-amber-200/60">
        <div className="flex items-baseline gap-1.5">
          <span className="font-mono text-[10px] text-amber-800/60">SC.</span>
          <h3 className="font-bold text-sm leading-tight text-zinc-900 line-clamp-1">
            {scene?.title || '—'}
          </h3>
        </div>
        <p className="text-[11px] leading-relaxed text-zinc-700 line-clamp-3 font-serif">
          {scene?.description || 'Awaiting director\'s vision…'}
        </p>
        {(isFallback || isMissing) && videoReady && (
          <div
            data-testid="scene-fallback-badge"
            className={`flex items-center gap-1 text-[10px] font-mono uppercase tracking-wider rounded-sm px-1.5 py-1 border ${
              isMissing
                ? 'bg-rose-100/80 text-rose-900 border-rose-300/80'
                : 'bg-amber-100/80 text-amber-900 border-amber-300/70'
            }`}
            title={
              isMissing
                ? 'No clip produced for this scene'
                : 'Real video generation was unavailable, so this scene falls back to a still image.'
            }
          >
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-current opacity-80" />
            {isMissing ? 'Missing clip' : 'Slide · fallback'}
            <span className="ml-auto text-[9px] opacity-60 normal-case tracking-normal">
              {isMissing ? '(retry required)' : '(no motion)'}
            </span>
          </div>
        )}
        {scene?.narration && (
          <div className="flex gap-1.5 text-[10.5px] leading-relaxed text-indigo-900/90 italic font-serif border-l-2 border-indigo-300/60 pl-1.5 pt-1">
            <Quote className="w-2.5 h-2.5 mt-0.5 flex-shrink-0 text-indigo-500/70" />
            <span className="line-clamp-2">{scene.narration}</span>
          </div>
        )}
        {scene?.shot_type && (
          <div className="flex items-center gap-1 text-[9px] font-mono uppercase tracking-wider text-amber-900/70 pt-1 border-t border-amber-200/60">
            <Camera className="w-2.5 h-2.5" />
            {scene.shot_type}
          </div>
        )}
      </div>
    </div>
  );
}

function StageProgress({ status }) {
  const activeIdx = STAGE_ORDER.indexOf(status);
  return (
    <div className="flex items-center gap-1.5 text-xs">
      {STAGES.map((s, i) => {
        const Icon = s.icon;
        const done = activeIdx > i;
        const active = activeIdx === i;
        return (
          <React.Fragment key={s.id}>
            <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full border transition-all ${
              active ? 'bg-indigo-600 border-indigo-400 text-white shadow-lg shadow-indigo-600/30' :
              done   ? 'bg-zinc-800 border-zinc-700 text-zinc-300' :
                       'bg-zinc-900 border-zinc-800 text-zinc-600'
            }`}>
              <Icon className={`w-3.5 h-3.5 ${active ? 'animate-pulse' : ''}`} />
              <span className="font-medium">{s.label}</span>
              {done && <CheckCircle2 className="w-3 h-3 text-emerald-400" />}
            </div>
            {i < STAGES.length - 1 && (
              <ChevronRight className={`w-3 h-3 ${done ? 'text-zinc-500' : 'text-zinc-800'}`} />
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
}

function EngineStack({ costBreakdown, totalUsd }) {
  // Always-on. Tells the user which real NVIDIA models are wired in
  // for this run.
  const items = [
    { name: 'Director (Script)',     model: 'meta/llama-3.3-70b-instruct', icon: Clapperboard, color: 'indigo' },
    { name: 'Prompt Engineer',       model: 'meta/llama-3.1-8b-instruct',  icon: Camera,       color: 'amber' },
    { name: 'Image Gen',             model: 'black-forest-labs/flux.1-dev (NIM)',  icon: ImageIcon, color: 'violet' },
    { name: 'Composer (Narrative)',  model: 'meta/llama-3.3-70b-instruct', icon: FileText,     color: 'emerald' },
    { name: 'Voice-Over',            model: 'nvidia/magpie-tts-multilingual (NVCF)',  icon: Mic,        color: 'rose' },
    { name: 'Music',                 model: 'procedural ambient pad (numpy + ffmpeg)',  icon: Music4,    color: 'cyan' },
  ];
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-[10px] font-mono uppercase tracking-widest text-zinc-500">
        <span>Engine · NVIDIA NIM</span>
        {totalUsd > 0 && (
          <span className="flex items-center gap-1.5 text-amber-300/90">
            <Coins className="w-3 h-3" />
            ≈ ${totalUsd.toFixed(4)} USD
          </span>
        )}
      </div>
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2">
        {items.map((it) => {
          const Icon = it.icon;
          return (
            <div key={it.name}
              className="flex flex-col gap-1 p-2.5 rounded-lg bg-zinc-900/60 border border-zinc-800/80 hover:border-zinc-700 transition">
              <div className="flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-widest text-zinc-400">
                <Icon className="w-3 h-3" />
                {it.name}
              </div>
              <div className="text-[10px] text-zinc-500 font-mono leading-tight break-all">
                {it.model}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Timeline({ scenes, durationSec }) {
  if (!scenes || scenes.length === 0) return null;
  let acc = 0;
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-[10px] font-mono uppercase tracking-widest text-zinc-500">
        <span>Timeline · 24 fps · Color</span>
        <span className="tabular-nums">{fmtTime(durationSec)}</span>
      </div>
      <div className="relative h-16 bg-zinc-900 rounded border border-zinc-800 overflow-hidden flex">
        {scenes.map((s, i) => {
          const w = (s.duration_sec / Math.max(1, durationSec)) * 100;
          const left = (acc / Math.max(1, durationSec)) * 100;
          acc += s.duration_sec;
          return (
            <div
              key={i}
              className="relative h-full border-r border-zinc-800 last:border-r-0"
              style={{ width: `${w}%` }}
            >
              {s.image_url && (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={s.image_url} alt="" className="absolute inset-0 w-full h-full object-cover opacity-60" />
              )}
              <div className="absolute inset-0 bg-gradient-to-r from-black/40 via-transparent to-black/40" />
              <div className="absolute bottom-0.5 left-1 right-1 text-[9px] font-mono text-amber-100 truncate">
                {String(i + 1).padStart(2, '0')} · {s.title}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function CinematicGenerator() {
  const {
    prompt, setPrompt, taskId, setTaskId, status, setStatus,
    scenes, setScenes, scenesCount, setScenesCount, storyboardCount, setStoryboardCount,
    videoUrlsCount, setVideoUrlsCount,
    finalVideoUrl, setFinalVideo,
    error, setError, title, setTitle, logline, setLogline, script, setScript,
    durationSec, setDurationSec, estimatedCostUsd, setEstimatedCost,
    costBreakdown, setCost, etaSeconds, setEtaSeconds, stageDurationsSec, setStageDurationsSec,
    scenesState, setScenesState,
    sceneProvenance, setSceneProvenance, renderQuality, setRenderQuality,
    warnings, setWarnings, retryPassCount, setRetryPassCount,
    style, setStyle, videoMode, setVideoMode, stylePresets, setStylePresets,
    reset,
  } = useVideoStore();

  // Default to 127.0.0.1 (IPv4) instead of `localhost` — Windows
  // resolves `localhost` to ::1 first on some hosts and the host
  // uvicorn binds only to IPv4, which makes the call fail at TCP
  // level with "Failed to fetch".
  const [apiBase] = useState(
    () => '/api',
  );
  // H1: no API key in the browser. Authentication is handled server-side
  // by the Next.js proxy, which adds X-Api-Key when forwarding to the backend.).
  const apiKey = '';

  const queryClient = useQueryClient();

  // Load available style presets once on mount.
  const stylesQuery = useQuery({
    queryKey: ['styles', apiBase],
    queryFn: async () => {
      const r = await fetch(`${apiBase}/styles`);
      if (!r.ok) throw new Error(`styles fetch failed: HTTP ${r.status}`);
      const j = await r.json();
      return j.presets || [];
    },
    staleTime: 60_000,
    retry: 1,
  });
  React.useEffect(() => {
    if (stylesQuery.data && stylesQuery.data.length) {
      setStylePresets(stylesQuery.data);
    }
  }, [stylesQuery.data, setStylePresets]);

  // Initial fetch via SSE. The SSE channel hands us a `snapshot` event
  // immediately on subscribe plus every `update` event thereafter — no
  // refetch loop needed. We keep a 6-second-safety-net poll as a fallback
  // so that if the EventSource silently dies on the server (proxy in
  // front of /api/outputs drops the keep-alive) the UI stays accurate.
  const taskQuery = useQuery({
    queryKey: ['task', taskId],
    queryFn: () => fetchTask(apiBase, taskId, apiKey),
    enabled: !!taskId,
    // SSE carries the live payload; only one slow-poll refresh lands
    // every 30 s as a safety net against silent SSE drops.
    refetchInterval: () => (TERMINAL.has(status) ? false : 30000),
    refetchIntervalInBackground: false,
    retry: 1,
  });

  // Live-stream updates when the task is in flight. The reducer below
  // trusts SSE for fast fields (status, count gauges, title) and falls
  // back to polling's full payload whenever SSE is silent.
  React.useEffect(() => {
    if (!taskId) return undefined;
    const apply = (data) => {
      if (!data) return;
      if (data.status) setStatus(data.status);
      if (data.type === 'snapshot') {
        // First frame after subscribe — populate lightweight labels now,
        // the full payload arrives later via SSE updates or polling.
        if (data.title) setTitle(data.title);
        setStoryboardCount(typeof data.storyboard_count === 'number' ? data.storyboard_count : 0);
        setVideoUrlsCount(typeof data.video_urls_count === 'number' ? data.video_urls_count : 0);
        if (data.final_video_url) setFinalVideo(data.final_video_url);
      }
      if (typeof data.scenes_count === 'number') setScenesCount(data.scenes_count);
      if (typeof data.storyboard_count === 'number') setStoryboardCount(data.storyboard_count);
      if (typeof data.video_urls_count === 'number') setVideoUrlsCount(data.video_urls_count);
      if (data.title) setTitle(data.title);
      if (data.logline) setLogline(data.logline);
      if (typeof data.duration_sec === 'number') setDurationSec(data.duration_sec);
      if (data.final_video_url) setFinalVideo(data.final_video_url);
      if (data.script) setScript(data.script);
      if (typeof data.estimated_cost_usd === 'number') setEstimatedCost(data.estimated_cost_usd);
      if (typeof data.eta_seconds === 'number') setEtaSeconds(data.eta_seconds);
      if (data.stage_durations_sec && typeof data.stage_durations_sec === 'object') {
        setStageDurationsSec(data.stage_durations_sec);
      }
      if (Array.isArray(data.scenes_state)) setScenesState(data.scenes_state);
      if (data.scene_provenance && typeof data.scene_provenance === 'object') {
        // Backend serialises int keys as strings; normalise here.
        const normalised = {};
        for (const [k, v] of Object.entries(data.scene_provenance)) {
          normalised[Number(k)] = v;
        }
        setSceneProvenance(normalised);
      }
      if (typeof data.render_quality === 'string') setRenderQuality(data.render_quality);
      if (Array.isArray(data.warnings)) setWarnings(data.warnings);
      if (typeof data.retry_pass_count === 'number') setRetryPassCount(data.retry_pass_count);
    };
    return subscribeTaskEvents(apiBase, taskId, apply, undefined, apiKey);
  }, [taskId, setStatus, setStoryboardCount, setVideoUrlsCount, setTitle, setLogline, setDurationSec, setFinalVideo, setScript, setEstimatedCost, setEtaSeconds, setStageDurationsSec, setScenesState]);


  React.useEffect(() => {
    const data = taskQuery.data;
    if (!data) return;
    if (data.status) setStatus(data.status);
    if (data.scenes) setScenes(data.scenes);
    if (data.title) setTitle(data.title);
    if (data.logline) setLogline(data.logline);
    if (typeof data.duration_sec === 'number') setDurationSec(data.duration_sec);
    if (data.final_video_url) setFinalVideo(data.final_video_url);
    // Only surface the `error` field when it actually belongs to the
    // task the user is currently looking at. Without this guard any
    // previous FAILED task that the polling fallback still has cached
    // would re-pollute the error bar on every redraw.
    if (data.error && data.id === taskId) setError(data.error);
    if (data.style) setStyle(data.style);
    if (typeof data.estimated_cost_usd === 'number' || data.cost_breakdown) {
      setCost(data.estimated_cost_usd ?? 0, data.cost_breakdown ?? {});
    }
    if (data.scene_provenance && typeof data.scene_provenance === 'object') {
      const normalised = {};
      for (const [k, v] of Object.entries(data.scene_provenance)) {
        normalised[Number(k)] = v;
      }
      setSceneProvenance(normalised);
    }
    if (typeof data.render_quality === 'string') setRenderQuality(data.render_quality);
    if (Array.isArray(data.warnings)) setWarnings(data.warnings);
    if (typeof data.retry_pass_count === 'number') setRetryPassCount(data.retry_pass_count);
    if (typeof data.video_mode === 'string') setVideoMode(data.video_mode);
  }, [taskQuery.data, setStatus, setScenes, setTitle, setLogline, setDurationSec, setFinalVideo, setError, setCost, setStyle, setSceneProvenance, setRenderQuality, setWarnings, setRetryPassCount, setVideoMode]);

  const generateMutation = useMutation({
    mutationFn: async ({ text, style, videoMode }) => {
      const headers = { 'Content-Type': 'application/json' };
      if (apiKey) headers['X-Api-Key'] = apiKey;
      const res = await fetch(`${apiBase}/generate`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ prompt: text, style, video_mode: videoMode }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status} ${await res.text().catch(() => '')}`);
      return res.json();
    },
    onMutate: () => setError(null),
    onSuccess: (data) => {
      setTaskId(data.task_id);
      setStatus(data.status || 'pending');
      setScenes([]);
      setTitle(null);
      setLogline(null);
      setDurationSec(0);
      setFinalVideo(null);
      setCost(0, {});
      setEtaSeconds(0);
      setStageDurationsSec({});
      if (data.style) setStyle(data.style);
      if (data.video_mode) setVideoMode(data.video_mode);
    },
    onError: (e) => setError(`Failed to initialize AI Director: ${e.message}`),
  });

  const isLoading = generateMutation.isPending;

  // Manual retry of selected scenes (only enabled when render_quality != ok).
  // Calls POST /api/tasks/<id>/retry-scene and lets the SSE stream
  // propagate the new provenance. UI never blocks on the response.
  const retryMutation = useMutation({
    mutationFn: async ({ taskId: tid, scenes: sceneList }) => {
      const headers = { 'Content-Type': 'application/json' };
      if (apiKey) headers['X-Api-Key'] = apiKey;
      const res = await fetch(`${apiBase}/tasks/${tid}/retry-scene`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ scenes: sceneList }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status} ${await res.text().catch(() => '')}`);
      return res.json();
    },
    onMutate: () => setError(null),
    onError: (e) => setError(`Retry failed: ${e.message}`),
  });

  // Scenes that still need a real video clip. Computed from provenance
  // once per status change so the button label / button list stays
  // stable while the user is reading the banner.
  const fallbackScenes = React.useMemo(() => {
    if (!renderQuality || renderQuality === 'ok') return [];
    const sp = sceneProvenance || {};
    return Object.keys(sp)
      .map((k) => Number(k))
      .filter((k) => sp[k] && sp[k] !== 'video')
      .sort((a, b) => a - b);
  }, [renderQuality, sceneProvenance]);
  const pollError = taskQuery.error && !TERMINAL.has(status);
  const hasStoryboard = scenes && scenes.length > 0;

  // Placeholder scene skeletons while the director is still working
  const placeholderCount = scenes?.length || scenesCount || 4;
  const placeholderScenes = Array.from({ length: placeholderCount }, (_, i) => scenes?.[i] || null);

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 font-sans selection:bg-indigo-500/40">
      {/* ===== Top nav ===== */}
      <nav className="px-4 py-3 sm:p-5 flex flex-wrap justify-between items-center gap-x-3 gap-y-2 border-b border-zinc-800 bg-zinc-950/80 backdrop-blur sticky top-0 z-50">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 shrink-0 bg-gradient-to-tr from-indigo-600 to-violet-400 rounded-lg flex items-center justify-center">
            <Clapperboard className="w-5 h-5 text-white" />
          </div>
          <span className="text-lg font-bold tracking-tight whitespace-nowrap shrink-0">VGO <span className="text-zinc-500 font-medium">Cinema</span></span>
        </div>
        <div className="flex min-w-0 items-center gap-2 sm:gap-4 text-sm">
          <span className="hidden sm:inline text-zinc-500">Storyboard</span>
          <span className="hidden sm:inline text-zinc-700">|</span>
          <button className="px-3 py-1.5 bg-zinc-100 text-zinc-900 rounded-full font-medium text-xs">Pro Account</button>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-4 py-10 space-y-10 sm:px-6">
        {/* ===== Hero / Input ===== */}
        {!taskId && (
          <section className="text-center pt-8 pb-2">
            <h1 className="text-4xl sm:text-5xl md:text-6xl font-extrabold text-balance mb-5 bg-gradient-to-b from-white to-zinc-400 bg-clip-text text-transparent tracking-tighter">
              Turn your vision into <br /> cinematic storyboards
            </h1>
            <p className="text-zinc-400 text-base sm:text-lg mb-10 max-w-2xl mx-auto">
              Our AI Director scripts, illustrates, and animates your idea into a shot-by-shot storyboard you can refine into a film.
            </p>
            <div className="relative max-w-3xl mx-auto group">
              <div className="absolute -inset-1 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-2xl blur opacity-25 group-hover:opacity-50 transition duration-1000"></div>
              <div className="relative bg-zinc-900 border border-zinc-800 rounded-2xl shadow-2xl">
                <div className="flex flex-col items-stretch gap-2 p-2 sm:flex-row sm:items-center sm:gap-2">
                  <input
                    className="min-w-0 w-full flex-1 bg-transparent border-none outline-none px-4 py-4 text-base text-white placeholder:text-zinc-600 sm:px-6 sm:text-lg"
                    placeholder="A futuristic neon city in the rain, cyberpunk aesthetic, wide cinematic shot..."
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && prompt && !isLoading && generateMutation.mutate({ text: prompt, style, videoMode })}
                  />
                  <button
                    onClick={() => generateMutation.mutate({ text: prompt, style, videoMode })}
                    disabled={isLoading || !prompt}
                    className="w-full shrink-0 justify-center px-5 py-3.5 bg-white text-zinc-900 rounded-xl font-bold hover:bg-zinc-200 transition-all flex items-center gap-2 disabled:opacity-50 sm:w-auto sm:px-7 sm:py-4"
                  >
                    {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Sparkles className="w-5 h-5" />}
                    Generate Storyboard
                  </button>
                </div>
                {/* Style preset chips (Cinematic / News / Documentary) */}
                <div className="px-4 pb-3 pt-1 flex flex-wrap items-center gap-2 border-t border-zinc-800/60">
                  <span className="text-[10px] font-mono uppercase tracking-widest text-zinc-500 mr-1">Style</span>
                  {(stylePresets.length ? stylePresets : [
                    { key: 'cinematic',   label: 'Cinematic' },
                    { key: 'news',        label: 'News' },
                    { key: 'documentary', label: 'Documentary' },
                  ]).map((p) => {
                    const active = style === p.key;
                    return (
                      <button
                        key={p.key}
                        type="button"
                        onClick={() => setStyle(p.key)}
                        className={
                          'px-3 py-1.5 rounded-full text-xs font-medium transition ' +
                          (active
                            ? 'bg-indigo-500/20 text-indigo-200 border border-indigo-400/50'
                            : 'bg-zinc-800/60 text-zinc-400 border border-zinc-700/50 hover:bg-zinc-800 hover:text-zinc-200')
                        }
                        title={p.description || p.label}
                      >
                        {p.label}
                      </button>
                    );
                  })}
                  {/* Video mode selector: Auto / Image-to-Video / Text-to-Video */}
                  <span className="text-[10px] font-mono uppercase tracking-widest text-zinc-500 ml-1 sm:ml-3 mr-1">Video</span>
                  {([
                    { key: 'auto',        label: 'Auto' },
                    { key: 'image2video', label: 'Image→Video' },
                    { key: 'text2video',  label: 'Text→Video' },
                  ]).map((m) => {
                    const active = videoMode === m.key;
                    return (
                      <button
                        key={m.key}
                        type="button"
                        onClick={() => setVideoMode(m.key)}
                        className={
                          'px-3 py-1.5 rounded-full text-xs font-medium transition ' +
                          (active
                            ? 'bg-emerald-500/20 text-emerald-200 border border-emerald-400/50'
                            : 'bg-zinc-800/60 text-zinc-400 border border-zinc-700/50 hover:bg-zinc-800 hover:text-zinc-200')
                        }
                        title={
                          m.key === 'auto'
                            ? 'Image-to-video when image is available, text-to-video otherwise'
                            : m.key === 'image2video'
                            ? 'Always use the generated image as reference (higher quality)'
                            : 'Prompt only, no image reference (no image dependency)'
                        }
                      >
                        {m.label}
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>
          </section>
        )}

        {/* ===== Stage progress ===== */}
        {taskId && (
          <section>
            <StageProgress status={status} />
          </section>
        )}

        {/* ===== Storyboard ===== */}
        {taskId && (
          <section>
            {/* Project header (slate) */}
            <div className="flex flex-wrap items-end justify-between gap-2 mb-5 pb-3 border-b border-zinc-800">
              <div>
                <div className="text-[10px] font-mono uppercase tracking-widest text-zinc-500 mb-1">
                  Project · {taskId?.slice(0, 8) || '—'}
                </div>
                <h2 className="text-3xl font-extrabold tracking-tight">
                  {title || (status === 'pending' || status === 'scripting'
                    ? <span className="text-zinc-600 animate-pulse">Director is writing the script…</span>
                    : 'Untitled')}
                </h2>
                {logline && (
                  <p className="text-zinc-400 mt-1.5 max-w-2xl italic font-serif">"{logline}"</p>
                )}
              </div>
              <div className="text-right font-mono text-[11px] uppercase tracking-widest text-zinc-500 space-y-0.5">
                <div>Style · <span className="text-zinc-300">{(style || 'cinematic').toUpperCase()}</span></div>
                <div>Mode · <span className="text-zinc-300">
                  {videoMode === 'image2video' ? 'IMAGE→VIDEO' : videoMode === 'text2video' ? 'TEXT→VIDEO' : 'AUTO'}
                </span></div>
                <div>Frames · <span className="text-zinc-300 tabular-nums">{scenes?.length || 0}</span></div>
                <div>Runtime · <span className="text-zinc-300 tabular-nums">{fmtTime(durationSec)}</span></div>
                <div>
                  ETA · <span className="text-amber-300/90 tabular-nums">
                    {etaSeconds === 0 || etaSeconds === undefined
                      ? (TERMINAL.has(status) ? '—' : '—')
                      : etaSeconds < 60
                        ? `~${Math.max(1, Math.floor(etaSeconds))}s`
                        : `~${Math.floor(etaSeconds / 60)}m ${Math.floor(etaSeconds % 60)}s`}
                  </span>
                </div>
                <div>Aspect · <span className="text-zinc-300">
                  {style === 'cinematic' ? '2.39 : 1' : style === 'documentary' ? '16 : 9' : '16 : 9'}
                </span></div>
              </div>
            </div>

            {/* Engine stack (always visible once a task exists) */}
            <div className="mb-6">
              <EngineStack
                costBreakdown={costBreakdown}
                totalUsd={estimatedCostUsd}
              />
            </div>

            {/* Film strip with perforated edges */}
            <div className="relative -mx-6 px-6 py-3 overflow-x-auto">
              {/* perforation top */}
              <div className="absolute top-0 left-0 right-0 h-2 bg-[length:24px_8px] bg-[linear-gradient(to_right,transparent_0,transparent_8px,zinc-900_8px,zinc-900_16px,transparent_16px,transparent_24px)] opacity-60 pointer-events-none" />
              <div className="flex gap-3 pb-3 pt-1">
                {placeholderScenes.map((s, i) => {
                  // Prefer per-scene SSE flags (live); fall back to scene
                  // fields once the polling refresh has painted them.
                  const ss = (scenesState || []).find((x) => x && x.scene === i + 1);
                  const imageReady = !!(ss && ss.image_ready) || !!s?.image_url || i < (storyboardCount || 0);
                  const videoReady = !!(ss && ss.video_ready) || !!s?.video_url || i < (videoUrlsCount || 0);
                  return (
                    <Frame
                      key={i}
                      index={i}
                      scene={s}
                      status={status}
                      sceneImageReady={imageReady}
                      sceneVideoReady={videoReady}
                      provenance={sceneProvenance?.[i + 1]}
                    />
                  );
                })}
              </div>
              {/* perforation bottom */}
              <div className="absolute bottom-0 left-0 right-0 h-2 bg-[length:24px_8px] bg-[linear-gradient(to_right,transparent_0,transparent_8px,zinc-900_8px,zinc-900_16px,transparent_16px,transparent_24px)] opacity-60 pointer-events-none" />
            </div>
          </section>
        )}

        {/* ===== Timeline ===== */}
        {hasStoryboard && (
          <section>
            <Timeline scenes={scenes} durationSec={durationSec} />
          </section>
        )}

        {/* ===== Final cut ===== */}
        {finalVideoUrl && (
          <section className="animate-in fade-in slide-in-from-bottom-4 duration-1000">
            <div className="flex items-center gap-2 mb-4">
              <FileText className="w-4 h-4 text-amber-400" />
              <h3 className="text-sm font-mono uppercase tracking-widest text-zinc-400">Director's Cut</h3>
            </div>
            {renderQuality && renderQuality !== 'ok' && (
              <div
                data-testid="render-quality-banner"
                role="status"
                aria-live="polite"
                className={`mb-4 px-4 py-3 rounded-lg border text-xs font-mono uppercase tracking-wider ${
                  (renderQuality === 'fallback_only' || renderQuality === 'missing')
                    ? 'bg-rose-500/10 border-rose-400/40 text-rose-200'
                    : 'bg-amber-500/10 border-amber-400/40 text-amber-200'
                }`}
              >
                <div className="flex items-start gap-3">
                  <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                  <div className="flex-1 leading-relaxed normal-case tracking-normal">
                    {renderQuality === 'fallback_only' ? (
                      <>
                        <strong className="block uppercase tracking-wider mb-1">Slideshow only</strong>
                        None of the scene videos reached the network camera — every clip
                        is a still image (Ken Burns). Likely cause: the upstream
                        video provider is rate-limited or not provisioned for your
                        account. Try a shorter prompt or wait a minute before
                        retrying.
                      </>
                    ) : renderQuality === 'missing' ? (
                      <>
                        <strong className="block uppercase tracking-wider mb-1">Missing scenes</strong>
                        One or more scenes produced neither a real video nor a
                        Ken Burns fallback — most likely a content-policy
                        rejection from the image / video provider. Retry the
                        missing scenes with a softer prompt.
                      </>
                    ) : (
                      <>
                        <strong className="block uppercase tracking-wider mb-1">Partial render</strong>
                        Some scenes used the still-image fallback (yellow
                        <span className="px-1 mx-1 bg-amber-200/20 rounded">Slide · fallback</span>
                        badge on the strip). The other scenes did produce real
                        motion. The MP4 still plays through, but motion coverage
                        isn't complete on this run.
                      </>
                    )}
                    {warnings && warnings.length > 0 && (
                      <ul className="mt-2 space-y-1 text-[11px] text-zinc-300/80">
                        {warnings.slice(-3).map((w, i) => (
                          <li key={i} className="font-mono normal-case">· {w}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                  {!retryMutation.isPending && fallbackScenes.length > 0 && renderQuality !== 'missing' && (
                    <button
                      type="button"
                      onClick={() =>
                        retryMutation.mutate({ taskId, scenes: fallbackScenes })
                      }
                      className="self-stretch px-3 py-2 rounded-md bg-amber-500/20 hover:bg-amber-500/30 border border-amber-400/40 text-amber-100 text-[11px] font-mono uppercase tracking-wider whitespace-nowrap"
                      title={`Queue a video retry for scenes ${fallbackScenes.join(', ')}`}
                    >
                      Retry {fallbackScenes.length} scene{fallbackScenes.length === 1 ? '' : 's'}
                    </button>
                  )}
                  {!retryMutation.isPending && renderQuality === 'missing' && (
                    <button
                      type="button"
                      onClick={reset}
                      className="self-stretch px-3 py-2 rounded-md bg-rose-500/20 hover:bg-rose-500/30 border border-rose-400/40 text-rose-100 text-[11px] font-mono uppercase tracking-wider whitespace-nowrap"
                      title="Restart the whole generation with a fresh prompt."
                    >
                      Restart
                    </button>
                  )}
                  {retryMutation.isPending && (
                    <span
                      className="self-stretch inline-flex items-center justify-center px-3 py-2 rounded-md bg-amber-500/10 border border-amber-400/40 text-amber-200 text-[11px] font-mono uppercase tracking-wider"
                    >
                      <Loader2 className="w-3 h-3 animate-spin mr-1.5" /> Retrying…
                    </span>
                  )}
                </div>
                {retryPassCount > 0 && (
                  <div className="mt-2 text-[10px] text-zinc-400/80 font-mono normal-case tracking-normal">
                    Retry passes attempted so far: {retryPassCount}
                  </div>
                )}
              </div>
            )}
            <div className="grid md:grid-cols-2 gap-6">
              <div className="aspect-video rounded-xl overflow-hidden ring-1 ring-white/10 shadow-2xl bg-zinc-900">
                {finalVideoUrl && isSafeMediaUrl(finalVideoUrl) && (
                finalVideoUrl?.endsWith('.mp4') ? (
                  <video
                    key={finalVideoUrl}
                    src={absoluteUrl(apiBase, finalVideoUrl)}
                    controls
                    autoPlay
                    loop
                    playsInline
                    className="w-full h-full object-cover"
                    onError={(e) => {
                      console.error('Video load failed:', finalVideoUrl, e);
                    }}
                  />
                ) : (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={finalVideoUrl} className="w-full h-full object-cover" alt="Final cut" />
                )
              )}
              {finalVideoUrl && !isSafeMediaUrl(finalVideoUrl) && (
                <div className="w-full h-full flex items-center justify-center text-zinc-500 text-sm">
                  Media URL rejected for security reasons.
                </div>
              )}
              </div>
              <div className="bg-amber-50 text-zinc-900 rounded-xl p-6 font-serif shadow-xl">
                <h4 className="font-bold text-lg mb-2">{title}</h4>
                <p className="text-sm leading-relaxed whitespace-pre-line text-zinc-800">
                  {(scenes || []).map((s, i) => `${i + 1}. ${s.title} — ${s.description}`).join('\n')}
                </p>
                <div className="flex gap-2 mt-5">
                  <a
                    href={isSafeMediaUrl(finalVideoUrl) ? absoluteUrl(apiBase, finalVideoUrl) : '#'}
                    download
                    className="px-4 py-2 bg-zinc-900 text-amber-50 rounded-lg text-xs font-mono uppercase tracking-wider"
                  >
                    Download Master
                  </a>
                  <button onClick={reset} className="px-4 py-2 border border-zinc-900 rounded-lg text-xs font-mono uppercase tracking-wider">
                    New Project
                  </button>
                </div>
              </div>
            </div>
          </section>
        )}

        {/* ===== Errors ===== */}
        {(error || pollError) && (
          <div className="flex items-center gap-3 p-4 bg-red-500/10 border border-red-500/20 text-red-400 rounded-xl text-sm">
            <AlertCircle className="w-5 h-5 flex-shrink-0" />
            <span>{error || 'Lost connection to the AI Director. Retrying…'}</span>
          </div>
        )}
      </main>
    </div>
  );
}
