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
// "Disagreement" = the models split on the VERDICT (some real, some
// fake). Spread in P(fake) magnitude among models that all land on the
// same side of 0.5 is NOT disagreement — they agree on the answer.
// A separate, disclosed near-boundary band flags genuinely low-margin
// unanimous calls.
const BOUNDARY_LO = 0.45; // disclosed band, not tuned
const BOUNDARY_HI = 0.55;

interface Decision {
  state: "real" | "fake" | "uncertain" | "none";
  reason: "consensus" | "split" | "borderline" | "none";
  pMean: number;
  nScored: number;
  votesReal: number;
  votesFake: number;
  dissentIds: string[];
}

function decide(models: ModelResult[]): Decision {
  const scored = models.filter(
    (m): m is ModelResult & { score: number } => m.score !== null,
  );
  const n = scored.length;
  if (n === 0) {
    return {
      state: "none",
      reason: "none",
      pMean: 0,
      nScored: 0,
      votesReal: 0,
      votesFake: 0,
      dissentIds: [],
    };
  }
  const ps = scored.map((m) => m.score);
  const pMean = ps.reduce((a, b) => a + b, 0) / n;
  const votesFake = ps.filter((p) => p >= 0.5).length;
  const votesReal = n - votesFake;
  const verdict: "real" | "fake" =
    votesFake > votesReal
      ? "fake"
      : votesReal > votesFake
        ? "real"
        : pMean >= 0.5
          ? "fake"
          : "real";
  const base = {
    pMean,
    nScored: n,
    votesReal,
    votesFake,
  };

  // Genuine disagreement: the verdicts split.
  if (votesFake > 0 && votesReal > 0) {
    const dissentIds = scored
      .filter((m) => (m.score >= 0.5 ? "fake" : "real") !== verdict)
      .map((m) => m.id);
    return { state: "uncertain", reason: "split", ...base, dissentIds };
  }
  // Unanimous verdict, but the ensemble sits on the decision boundary.
  if (pMean >= BOUNDARY_LO && pMean <= BOUNDARY_HI) {
    return { state: "uncertain", reason: "borderline", ...base, dissentIds: [] };
  }
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
  const frameNote = isVideo ? ` · mean-pooled over ${r.n_frames ?? 0} frames` : "";

  return (
    <div className="results">
      {/* 03 — input transformed into output */}
      <div>
        <Head
          idx="03"
          title="Input → transform"
          meta={isVideo ? `video · ${r.n_frames ?? 0} frames sampled` : "settled manifolds"}
        />
        <div className="io">
          <figure className="specimen">
            <img src={r.input} alt="model input — face crop" />
            <figcaption className="cap">
              <b>{isVideo ? "Most suspicious frame" : "Input"}</b> · 256² crop{" "}
              {r.face_detected ? (
                <span>· face detected</span>
              ) : (
                <span className="warn">· no face — centre crop</span>
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
        {d.state === "none" && (
          <div className="peak none">
            <div className="peak-none">
              No calibrated score — only the heuristic (diagnostic-only) was run.
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
                    : "Near decision boundary"}
                </div>
                <div className="peak-model">
                  {d.reason === "split" ? (
                    <>
                      {d.votesReal} real / {d.votesFake} fake · mean P(fake){" "}
                      {pct(d.pMean)}
                      {frameNote}. The branches split on the verdict —
                      single-model verdicts are not trustworthy here; see
                      the per-model breakdown above.
                    </>
                  ) : (
                    <>
                      All {d.nScored} models agree, but the ensemble mean
                      P(fake) {pct(d.pMean)} sits on the decision boundary
                      {frameNote}. Treat this as a low-confidence call.
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
                  {d.state === "fake" ? d.votesFake : d.votesReal} of{" "}
                  {d.nScored} models
                </div>
                <div className="peak-model">
                  mean P(fake) {pct(d.pMean)} · the models agree this{" "}
                  {isVideo ? "video" : "image"} is{" "}
                  <b>{d.state === "fake" ? "a deepfake" : "real"}</b>
                  {frameNote}
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
      </div>
    </div>
  );
}
