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
}

export interface Peak {
  model_id: string;
  model_label: string;
  score: number;
  verdict: "fake" | "real";
}

export interface DetectResults {
  input: string;
  face_detected: boolean;
  peak: Peak | null;
  models: ModelResult[];
}

export interface DetectResponse {
  filename: string | null;
  results: DetectResults;
}

export async function fetchModels(): Promise<ModelInfo[]> {
  const res = await fetch("/api/models");
  if (!res.ok) throw new Error(`GET /api/models failed: ${res.status}`);
  return (await res.json()) as ModelInfo[];
}

export async function detect(
  file: File,
  modelIds: string[],
): Promise<DetectResponse> {
  const form = new FormData();
  form.append("file", file);
  for (const id of modelIds) form.append("model_ids", id);

  const res = await fetch("/api/detect", { method: "POST", body: form });
  const body = (await res.json()) as DetectResponse & { error?: string };
  if (!res.ok) {
    throw new Error(body.error ?? `POST /api/detect failed: ${res.status}`);
  }
  return body;
}
