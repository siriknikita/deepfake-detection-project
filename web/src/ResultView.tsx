import type { DetectResults, ModelResult } from "./api";

function pct(x: number): string {
  return (x * 100).toFixed(1) + "%";
}

// Confidence in the model's OWN verdict: 0.5 → 0% (coin flip),
// 0.007 or 0.993 → ~99% (decisive, either way).
function confidence(score: number): number {
  return Math.max(score, 1 - score);
}

// Per-frame P(fake) over the sampled video frames.
function Sparkline({ values, fake }: { values: number[]; fake: boolean }) {
  const w = 104;
  const h = 24;
  const n = values.length;
  const pts = values
    .map((v, i) => {
      const x = n === 1 ? w / 2 : (i / (n - 1)) * w;
      const y = (1 - Math.min(1, Math.max(0, v))) * h;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg className={`spark ${fake ? "fake" : "real"}`} viewBox={`0 0 ${w} ${h}`}>
      <line x1="0" y1={h / 2} x2={w} y2={h / 2} className="spark-mid" />
      <polyline points={pts} />
    </svg>
  );
}

// Disagreement-aware fusion. Deliberately unweighted and pre-stated:
// hand-tuned ensemble weights chosen from a few in-the-wild images are
// not defensible. The per-model table remains the full diagnostic view;
// this only decides the headline.
//
// A hard 0.5 cut is brittle: P(fake) 0.549 and 0.999 are not the same
// vote. Each model is classified with a disclosed dead-band — inside
// [0.45, 0.55] the model is *undecided* (near-chance, abstains), not a
// dissenter. "Disagreement" then means the CONFIDENT branches split
// (some confidently real, some confidently fake); a near-chance model
// no longer trips it.
const BOUNDARY_LO = 0.45; // disclosed dead-band, not tuned
const BOUNDARY_HI = 0.55;

interface Decision {
  state: "real" | "fake" | "uncertain" | "none";
  reason: "consensus" | "split" | "borderline" | "none";
  pMean: number;
  nScored: number;
  confReal: number;
  confFake: number;
  undecided: number;
  undecidedLabels: string[];
  dissentIds: string[];
}

function decide(models: ModelResult[]): Decision {
  const scored = models.filter(
    (m): m is ModelResult & { score: number } => m.score !== null,
  );
  const n = scored.length;
  const empty = {
    pMean: 0,
    nScored: n,
    confReal: 0,
    confFake: 0,
    undecided: 0,
    undecidedLabels: [] as string[],
    dissentIds: [] as string[],
  };
  if (n === 0) return { state: "none", reason: "none", ...empty };

  const pMean = scored.reduce((a, m) => a + m.score, 0) / n;
  const vote = (s: number): "real" | "fake" | "undecided" =>
    s >= BOUNDARY_HI ? "fake" : s <= BOUNDARY_LO ? "real" : "undecided";

  const confFakeIds = scored.filter((m) => vote(m.score) === "fake");
  const confRealIds = scored.filter((m) => vote(m.score) === "real");
  const undecidedM = scored.filter((m) => vote(m.score) === "undecided");
  const base = {
    pMean,
    nScored: n,
    confReal: confRealIds.length,
    confFake: confFakeIds.length,
    undecided: undecidedM.length,
    undecidedLabels: undecidedM.map((m) => m.label),
  };

  // Confident branches split on the verdict — genuine disagreement.
  if (confFakeIds.length > 0 && confRealIds.length > 0) {
    const minoritySide =
      confFakeIds.length <= confRealIds.length ? confFakeIds : confRealIds;
    return {
      state: "uncertain",
      reason: "split",
      ...base,
      dissentIds: minoritySide.map((m) => m.id),
    };
  }
  // Nobody commits — the whole ensemble sits in the indecision band.
  if (confFakeIds.length === 0 && confRealIds.length === 0) {
    return { state: "uncertain", reason: "borderline", ...base, dissentIds: [] };
  }
  // One confident side, the rest (if any) undecided → consensus of the
  // committed models.
  const verdict = confFakeIds.length > 0 ? "fake" : "real";
  return { state: verdict, reason: "consensus", ...base, dissentIds: [] };
}

function Head({ idx, title, meta }: { idx: string; title: string; meta: string }) {
  return (
    <div className="sec-head">
      <span className="sec-idx">{idx}</span>
      <span className="sec-title">{title}</span>
      <span className="sec-rule" />
      <span className="sec-meta">{meta}</span>
    </div>
  );
}

function Transform({ model }: { model: ModelResult }) {
  return (
    <div className="xform">
      <div className="xform-head">
        <span className="xform-id">{model.label}</span>
        <span className="xform-tag">
          {model.kind === "heuristic"
            ? "manifold settlement"
            : `${model.channels.length} input channels`}
        </span>
      </div>

      {model.panel && (
        <figure className="plate">
          <img src={model.panel} alt={`${model.label} 6-panel manifold`} />
          <figcaption className="cap">
            6-panel: input · W_cnn · z_forged · z* · R · cracks
          </figcaption>
        </figure>
      )}

      {model.channels.length > 0 && (
        <div className="strip">
          {model.channels.map((c) => (
            <figure key={c.label} className="cell">
              <img src={c.png} alt={c.label} />
              <figcaption>{c.label}</figcaption>
            </figure>
          ))}
        </div>
      )}
    </div>
  );
}

export default function ResultView({ r }: { r: DetectResults }) {
  const d = decide(r.models);
  const dissent = new Set(d.dissentIds);
  const isVideo = r.kind === "video";
  const face = r.face;
  const faceUnusable = !!face && !face.usable;
  const usedFrames = isVideo
    ? (face?.usable_frames ?? r.n_frames ?? 0)
    : 0;
  const frameNote = isVideo
    ? ` · mean-pooled over ${usedFrames} face-valid frame${usedFrames === 1 ? "" : "s"}`
    : "";
  // Strongest single-frame signal across models (paper's max-pool axis).
  const maxScores = r.models
    .map((m) => m.score_max)
    .filter((x): x is number => typeof x === "number");
  const maxAny = maxScores.length ? Math.max(...maxScores) : null;
  const maxNote =
    isVideo && maxAny !== null
      ? ` · max-pool reached ${pct(maxAny)} on a single frame`
      : "";

  return (
    <div className="results">
      {/* 03 — input transformed into output */}
      <div>
        <Head
          idx="03"
          title="Input → transform"
          meta={
            isVideo
              ? `video · ${usedFrames}/${face?.total_frames ?? r.n_frames ?? 0} usable frames`
              : "settled manifolds"
          }
        />
        <div className="io">
          <figure className="specimen">
            <img src={r.input} alt="model input — face crop" />
            <figcaption className="cap">
              <b>{isVideo ? "Representative frame" : "Input"}</b> · 256² crop{" "}
              {isVideo ? (
                <span className={faceUnusable ? "warn" : ""}>
                  · {usedFrames}/{face?.total_frames ?? r.n_frames ?? 0} frames
                  with a usable face
                </span>
              ) : faceUnusable ? (
                <span className="warn">
                  · no usable face
                  {face?.prob != null ? ` (p=${face.prob.toFixed(2)})` : ""}
                </span>
              ) : (
                <span>
                  · face detected
                  {face?.prob != null ? ` (p=${face.prob.toFixed(2)})` : ""}
                </span>
              )}
            </figcaption>
          </figure>
          <div className="io-link">→</div>
          <div className="io-out">
            {r.models.map((m) => (
              <Transform key={m.id} model={m} />
            ))}
          </div>
        </div>
      </div>

      {/* 04 — per-model breakdown */}
      <div>
        <Head idx="04" title="Model breakdown" meta="confidence in verdict" />
        <div className="panel">
          <table className="grid-table">
            <thead>
              <tr>
                <th style={{ width: "34%" }}>Model</th>
                <th>Verdict</th>
                <th style={{ width: "30%" }}>Confidence</th>
                <th>P(fake)</th>
              </tr>
            </thead>
            <tbody>
              {r.models.map((m) => {
                const isDissent = d.state === "uncertain" && dissent.has(m.id);
                const conf = m.score === null ? null : confidence(m.score);
                return (
                  <tr
                    key={m.id}
                    className={isDissent ? "is-peak dissent" : ""}
                  >
                    <td className="m-name">
                      <span>{m.label}</span>
                      {m.frame_scores && m.frame_scores.length > 1 && (
                        <Sparkline
                          values={m.frame_scores}
                          fake={m.verdict === "fake"}
                        />
                      )}
                    </td>
                    <td>
                      {m.verdict === null ? (
                        <span className="tag diag">diagnostic</span>
                      ) : (
                        <span className={`tag ${m.verdict}`}>
                          {m.verdict.toUpperCase()}
                        </span>
                      )}
                    </td>
                    <td>
                      {conf === null ? (
                        <span className="num dim">no calibrated score</span>
                      ) : (
                        <div className="conf">
                          <div className={`gauge ${m.verdict}`}>
                            <i style={{ width: `${Math.max(2, conf * 100)}%` }} />
                          </div>
                          <span className="conf-num">{pct(conf)}</span>
                        </div>
                      )}
                    </td>
                    <td className={`num ${m.score === null ? "dim" : ""}`}>
                      {m.score === null ? "—" : pct(m.score)}
                      {m.score !== null && m.score_max !== undefined && (
                        <span className="sub">
                          mean · max {pct(m.score_max)}
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* 05 — summary */}
      <div>
        <Head
          idx="05"
          title="Summary"
          meta={isVideo ? "consensus · pooled" : "model consensus"}
        />
        {faceUnusable ? (
          <div className="peak none">
            <div className="peak-none">
              {isVideo
                ? `No usable face — only ${face?.usable_frames ?? 0} of ${
                    face?.total_frames ?? 0
                  } sampled frames had a face above the ${pct(
                    face?.min_prob ?? 0.9,
                  )} detector threshold.`
                : `No usable face — detector confidence ${
                    face?.prob != null ? pct(face.prob) : "low"
                  } is below the ${pct(face?.min_prob ?? 0.9)} threshold.`}{" "}
              This input is outside the face detector's domain — the
              per-model scores above are not a meaningful verdict.
            </div>
          </div>
        ) : (
          <>
            {d.state === "none" && (
              <div className="peak none">
                <div className="peak-none">
                  No calibrated score — only the heuristic (diagnostic-only)
                  was run.
                </div>
              </div>
            )}
        {d.state === "uncertain" && (
          <div className="peak uncertain">
            <div className="peak-grid">
              <div>
                <div className="peak-label">
                  {d.reason === "split"
                    ? "Models disagree"
                    : "No branch commits"}
                </div>
                <div className="peak-model">
                  {d.reason === "split" ? (
                    <>
                      {d.confReal} confidently real / {d.confFake} confidently
                      fake
                      {d.undecided > 0
                        ? ` · ${d.undecided} undecided`
                        : ""}{" "}
                      · mean P(fake) {pct(d.pMean)}
                      {frameNote}
                      {maxNote}. The confident branches split on the
                      verdict — single-model verdicts are not trustworthy
                      here; see the per-model breakdown above.
                    </>
                  ) : (
                    <>
                      All {d.nScored} models fall inside the{" "}
                      {pct(BOUNDARY_LO)}–{pct(BOUNDARY_HI)} indecision band
                      (mean P(fake) {pct(d.pMean)}){frameNote}. No branch
                      commits — treat as undecided.
                    </>
                  )}
                </div>
              </div>
              <div className="peak-score uncertain">{pct(d.pMean)}</div>
              <div className="peak-verdict uncertain">UNCERTAIN</div>
            </div>
          </div>
        )}
        {(d.state === "real" || d.state === "fake") && (
          <div className={`peak ${d.state}`}>
            <div className="peak-grid">
              <div>
                <div className="peak-label">
                  Consensus ·{" "}
                  {d.state === "fake" ? d.confFake : d.confReal} of{" "}
                  {d.nScored} models
                  {d.undecided > 0
                    ? ` (${d.undecidedLabels.join(", ")} undecided)`
                    : ""}
                </div>
                <div className="peak-model">
                  mean P(fake) {pct(d.pMean)} · the committed models lean{" "}
                  <b>{d.state === "fake" ? "fake" : "real"}</b> at the
                  provisional 0.5 threshold
                  {frameNote}
                  {maxNote}
                </div>
              </div>
              <div className={`peak-score ${d.state}`}>
                {pct(Math.max(d.pMean, 1 - d.pMean))}
              </div>
              <div className={`peak-verdict ${d.state}`}>
                {d.state.toUpperCase()}
              </div>
            </div>
          </div>
            )}
          </>
        )}
        {!faceUnusable && d.state !== "none" && (
          <p className="peak-caveat">
            Scores are uncalibrated ranker outputs; the 0.5 cut is a
            provisional placeholder, not a validated decision threshold.
            The absolute % is not a deepfake probability — only relative
            comparison across models and rank / AUROC are meaningful.
          </p>
        )}
      </div>
    </div>
  );
}
