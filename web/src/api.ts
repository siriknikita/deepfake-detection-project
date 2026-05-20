// Typed client for the FastAPI serving layer. All paths are relative
// so the Vite dev proxy (and a same-origin deploy) just work.

export interface ModelInfo {
  id: string;
  label: string;
  description: string;
  kind: "heuristic" | "cnn";
  in_channels: number;
  available: boolean;
}

export interface ChannelPreview {
  label: string;
  png: string;
}

export interface ModelResult {
  id: string;
  label: string;
  kind: "heuristic" | "cnn";
  score: number | null;
  verdict: "fake" | "real" | null;
  note: string | null;
  panel: string | null;
  channels: ChannelPreview[];
  frame_scores?: number[]; // video only — per-frame P(fake) over usable frames
  score_max?: number; // video only — max-pool P(fake)
}

export interface FaceInfo {
  detected: boolean;
  usable: boolean;
  prob?: number; // image
  usable_frames?: number; // video
  total_frames?: number; // video
  min_prob?: number;
}

export interface DetectResults {
  input: string;
  face_detected: boolean;
  face?: FaceInfo;
  kind: "image" | "video";
  n_frames?: number; // video only
  models: ModelResult[];
}

export interface Progress {
  status: string;
  done: number;
  total: number;
}

export async function fetchModels(): Promise<ModelInfo[]> {
  const res = await fetch("/api/models");
  if (!res.ok) throw new Error(`GET /api/models failed: ${res.status}`);
  return (await res.json()) as ModelInfo[];
}

interface DetectBody {
  results?: DetectResults;
  job_id?: string;
  error?: string;
}

interface JobBody {
  status: string;
  done: number;
  total: number;
  result: DetectResults | null;
  error: string | null;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Submit a file for detection. Images resolve inline; videos are polled
 * as a background job, reporting progress via `onProgress`.
 */
export async function detect(
  file: File,
  modelIds: string[],
  onProgress?: (p: Progress) => void,
): Promise<DetectResults> {
  const form = new FormData();
  form.append("file", file);
  for (const id of modelIds) form.append("model_ids", id);

  const res = await fetch("/api/detect", { method: "POST", body: form });
  const body = (await res.json()) as DetectBody;
  if (!res.ok) {
    throw new Error(body.error ?? `POST /api/detect failed: ${res.status}`);
  }
  if (body.results) return body.results; // image — inline

  const jobId = body.job_id;
  if (!jobId) throw new Error("malformed response: no results and no job_id");

  // Video — poll until done or error.
  for (;;) {
    await sleep(900);
    const jr = await fetch(`/api/jobs/${jobId}`);
    const job = (await jr.json()) as JobBody;
    if (!jr.ok) throw new Error(`job poll failed: ${jr.status}`);
    onProgress?.({ status: job.status, done: job.done, total: job.total });
    if (job.status === "done" && job.result) return job.result;
    if (job.status === "error") {
      throw new Error(job.error ?? "video job failed");
    }
  }
}
