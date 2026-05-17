import { useEffect, useRef, useState } from "react";
import {
  detect,
  fetchModels,
  type DetectResults,
  type ModelInfo,
  type Progress,
} from "./api";
import ResultView from "./ResultView";

function metaTag(m: ModelInfo): string {
  if (m.kind === "heuristic") return "math pipeline";
  if (m.in_channels >= 9) return "9-ch · rgb+phys+freq";
  if (m.in_channels >= 6) return "6-ch · rgb+phys";
  return "3-ch · rgb";
}

export default function App() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DetectResults | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetchModels()
      .then((m) => {
        setModels(m);
        // Pre-select the heuristic model — it always works on a fresh clone.
        setSelected(new Set(m.filter((x) => x.id === "heuristic").map((x) => x.id)));
      })
      .catch((e: unknown) => setError(String(e)));
  }, []);

  function toggle(id: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  async function run() {
    if (!file || selected.size === 0) return;
    setBusy(true);
    setError(null);
    setResult(null);
    setProgress(null);
    try {
      const r = await detect(file, [...selected], setProgress);
      setResult(r);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      setProgress(null);
    }
  }

  function busyLabel(): string {
    if (progress && progress.total > 0) {
      return `Analysing frame ${progress.done} / ${progress.total}`;
    }
    return progress ? "Decoding video…" : "Analysing…";
  }

  const armed = models.filter((m) => selected.has(m.id)).length;

  return (
    <div className="page">
      <header className="masthead">
        <div>
          <h1 className="wordmark">
            Hyperplane<span className="forge">-Forge</span>
          </h1>
          <p className="tagline">Multi-model deepfake detection</p>
        </div>
        <div className="readout">
          <div><span className="live">●</span> service online</div>
          <div>{models.length} models loaded</div>
        </div>
      </header>

      {/* 01 — models */}
      <div className="sec-head">
        <span className="sec-idx">01</span>
        <span className="sec-title">Select models</span>
        <span className="sec-rule" />
        <span className="sec-meta">{armed} of {models.length} selected</span>
      </div>
      <div className="panel">
        <div className="models">
          {models.map((m) => {
            const on = selected.has(m.id);
            return (
              <label
                key={m.id}
                className={`model ${on ? "on" : ""} ${m.available ? "" : "dead"}`}
                title={m.available ? m.description : "Checkpoint not found on this checkout"}
              >
                <input
                  type="checkbox"
                  checked={on}
                  disabled={!m.available}
                  onChange={() => toggle(m.id)}
                />
                <span className="box" aria-hidden="true" />
                <span className="m-text">
                  <span className="m-label">{m.label}</span>
                  <span className="m-desc">{m.description}</span>
                </span>
                <span className="m-meta">{m.available ? metaTag(m) : "unavailable"}</span>
              </label>
            );
          })}
          {models.length === 0 && !error && (
            <div className="loading">Linking model registry</div>
          )}
        </div>
      </div>

      {/* 02 — input */}
      <div className="sec-head">
        <span className="sec-idx">02</span>
        <span className="sec-title">Provide input</span>
        <span className="sec-rule" />
        <span className="sec-meta">image or video</span>
      </div>
      <label
        className={`intake ${dragging ? "drag" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const f = e.dataTransfer.files?.[0];
          if (f) setFile(f);
        }}
      >
        <input
          ref={fileRef}
          type="file"
          accept="image/*,video/*"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        {file ? (
          <div className="loaded">{file.name}</div>
        ) : (
          <>
            <div className="lead">Drop a file, or click to browse</div>
            <div className="hint">image or video — analysed on the largest detected face</div>
          </>
        )}
      </label>

      <button
        className="execute"
        disabled={busy || !file || selected.size === 0}
        onClick={run}
      >
        {busy
          ? busyLabel()
          : `Run ${selected.size} model${selected.size === 1 ? "" : "s"}`}
      </button>

      {error && <div className="fault">{error}</div>}

      {result != null && <ResultView r={result} />}
    </div>
  );
}
