import { create } from 'zustand';

export type VideoStatus =
  | 'idle'
  | 'pending'
  | 'scripting'
  | 'image_gen'
  | 'video_gen'
  | 'composing'
  | 'completed'
  | 'failed';

export interface Scene {
  scene: number;
  title: string;
  description: string;
  shot_type: string;
  lighting: string;
  image_prompt?: string | null;
  image_url?: string | null;
  video_url?: string | null;
  motion?: string | null;
  duration_sec: number;
  dialogue?: string | null;
  sfx?: string | null;
  narration?: string | null;
  audio_path?: string | null;
}

export interface SceneState {
  scene: number;
  image_ready: boolean;
  video_ready: boolean;
  audio_ready: boolean;
}

export interface CostBreakdown {
  director_llm?: number;
  prompt_enhancer_llm?: number;
  composer_llm?: number;
  image_gen?: number;
  tts?: number;
  video_compose?: number;
}

export type VideoStyle = 'cinematic' | 'news' | 'documentary' | string;

// Video generation mode. "auto" uses image-to-video when an image is
// available, "image2video" always passes the image, "text2video" uses
// the prompt only.
export type VideoMode = 'auto' | 'image2video' | 'text2video';

// Per-scene provenance. "video" = real upstream video clip; "kenburns"
// = single image stretched to fill the slot (fallback when the video
// provider rejected / rate-limited us); "missing" = no clip at all.
export type SceneProvenance = 'video' | 'kenburns' | 'missing';

// Coarse pipeline health roll-up surfaced by the backend.
export type RenderQuality = 'ok' | 'partial' | 'fallback_only' | string;

export interface StylePreset {
  key: VideoStyle;
  label: string;
  description: string;
  aspect: string;
}

interface VideoTaskState {
  taskId: string | null;
  prompt: string;
  status: VideoStatus;
  scenes: Scene[];
  // Lightweight gauges updated by SSE so the storyboard grid can show
  // live progress without re-fetching the full payload every 6 seconds.
  scenesCount: number;
  storyboardCount: number;
  videoUrlsCount: number;
  title: string | null;
  logline: string | null;
  script: string;
  durationSec: number;
  finalVideoUrl: string | null;
  error: string | null;
  estimatedCostUsd: number;
  costBreakdown: CostBreakdown;
  // Rolling ETA, in seconds, fed by the SSE payload. 0 means idle.
  etaSeconds: number;
  // Per-stage historical timings (seconds) accumulated from past runs.
  stageDurationsSec: Record<string, number>;
  // Per-scene readiness flags from SSE so the storyboard grid animates
  // each scene chip independently as the chord fans out 4 sub-tasks.
  scenesState: SceneState[];
  // Per-scene provenance (1-indexed scene number → real|fake|missing).
  sceneProvenance: Record<number, SceneProvenance>;
  // Pipeline health roll-up — drives the "Render quality" banner.
  renderQuality: RenderQuality;
  // Rolling list of advisory / recovery messages. Persisted server-side;
  // the UI can show them under the banner.
  warnings: string[];
  // How many retry passes have been attempted (auto + manual). 0 = none yet.
  retryPassCount: number;
  style: VideoStyle;
  videoMode: VideoMode;
  stylePresets: StylePreset[];
  setPrompt: (prompt: string) => void;
  setTaskId: (id: string | null) => void;
  setStatus: (status: VideoStatus) => void;
  setScenes: (scenes: Scene[]) => void;
  setScenesCount: (n: number) => void;
  setStoryboardCount: (n: number) => void;
  setVideoUrlsCount: (n: number) => void;
  setTitle: (title: string | null) => void;
  setLogline: (logline: string | null) => void;
  setScript: (script: string) => void;
  setDurationSec: (sec: number) => void;
  setFinalVideo: (url: string | null) => void;
  setError: (error: string | null) => void;
  setEstimatedCost: (usd: number) => void;
  setCost: (cost: number, breakdown: CostBreakdown) => void;
  setEtaSeconds: (sec: number) => void;
  setStageDurationsSec: (m: Record<string, number>) => void;
  setScenesState: (s: SceneState[]) => void;
  setSceneProvenance: (m: Record<number, SceneProvenance>) => void;
  setRenderQuality: (q: RenderQuality) => void;
  setWarnings: (w: string[]) => void;
  setRetryPassCount: (n: number) => void;
  setStyle: (style: VideoStyle) => void;
  setVideoMode: (mode: VideoMode) => void;
  setStylePresets: (presets: StylePreset[]) => void;
  reset: () => void;
}

export const useVideoStore = create<VideoTaskState>((set) => ({
  taskId: null,
  prompt: '',
  status: 'idle',
  scenes: [],
  scenesCount: 0,
  storyboardCount: 0,
  videoUrlsCount: 0,
  title: null,
  logline: null,
  script: '',
  durationSec: 0,
  finalVideoUrl: null,
  error: null,
  estimatedCostUsd: 0,
  costBreakdown: {},
  etaSeconds: 0,
  stageDurationsSec: {},
  scenesState: [],
  sceneProvenance: {},
  renderQuality: 'ok',
  warnings: [],
  retryPassCount: 0,
  style: 'cinematic',
  videoMode: 'auto',
  stylePresets: [],
  setPrompt: (prompt) => set({ prompt }),
  setTaskId: (taskId) => set({ taskId }),
  setStatus: (status) => set({ status }),
  setScenes: (scenes) => set({ scenes }),
  setScenesCount: (scenesCount) => set({ scenesCount }),
  setStoryboardCount: (storyboardCount) => set({ storyboardCount }),
  setVideoUrlsCount: (videoUrlsCount) => set({ videoUrlsCount }),
  setTitle: (title) => set({ title }),
  setLogline: (logline) => set({ logline }),
  setScript: (script) => set({ script }),
  setDurationSec: (durationSec) => set({ durationSec }),
  setFinalVideo: (finalVideoUrl) => set({ finalVideoUrl }),
  setError: (error) => set({ error }),
  setEstimatedCost: (estimatedCostUsd) => set({ estimatedCostUsd }),
  setCost: (estimatedCostUsd, costBreakdown) => set({ estimatedCostUsd, costBreakdown }),
  setEtaSeconds: (etaSeconds) => set({ etaSeconds }),
  setStageDurationsSec: (stageDurationsSec) => set({ stageDurationsSec }),
  setScenesState: (scenesState) => set({ scenesState }),
  setSceneProvenance: (sceneProvenance) => set({ sceneProvenance }),
  setRenderQuality: (renderQuality) => set({ renderQuality }),
  setWarnings: (warnings) => set({ warnings }),
  setRetryPassCount: (retryPassCount) => set({ retryPassCount }),
  setStyle: (style) => set({ style }),
  setVideoMode: (videoMode) => set({ videoMode }),
  setStylePresets: (stylePresets) => set({ stylePresets }),
  reset: () => set({
    taskId: null, prompt: '', status: 'idle', scenes: [],
    scenesCount: 0, storyboardCount: 0, videoUrlsCount: 0,
    title: null, logline: null, script: '',
    durationSec: 0, finalVideoUrl: null, error: null,
    estimatedCostUsd: 0, costBreakdown: {},
    etaSeconds: 0, stageDurationsSec: {}, scenesState: [],
    sceneProvenance: {}, renderQuality: 'ok', warnings: [],
    retryPassCount: 0,
    style: 'cinematic',
    videoMode: 'auto',
  }),
}));
