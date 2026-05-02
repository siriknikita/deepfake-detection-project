// Hyperplane-Forge — diploma paper
//
// Compile with:
//   typst compile paper/main.typ paper/main.pdf
// Or live-preview:
//   typst watch paper/main.typ paper/main.pdf
//
// The paper is split into per-phase section files under paper/sections/ so
// each algorithm stage has its own self-contained source. main.typ stitches
// them together and owns global typographic settings.

#set document(
  title: "Hyperplane-Forge: Deepfake Detection via Physical-Manifold Settlement",
  author: "Siryk Mykyta",
)

#set page(
  paper: "a4",
  margin: (x: 2.2cm, y: 2.4cm),
  numbering: "1",
  number-align: center,
)

#set text(
  font: "New Computer Modern",
  size: 11pt,
  lang: "en",
)

#set par(justify: true, leading: 0.62em, first-line-indent: 1em)

#set heading(numbering: "1.1")

#show heading.where(level: 1): it => {
  pagebreak(weak: true)
  v(1em)
  text(size: 18pt, weight: "bold")[#it]
  v(0.5em)
}

#show heading.where(level: 2): it => {
  v(0.8em)
  text(size: 13pt, weight: "bold")[#it]
  v(0.2em)
}

#set math.equation(numbering: "(1)")

#show math.equation: set text(font: "New Computer Modern Math")

// ---------- Title page ----------

#align(center)[
  #v(3cm)
  #text(size: 22pt, weight: "bold")[Hyperplane-Forge]
  #v(0.4em)
  #text(size: 14pt)[
    Deepfake Detection via Physical-Manifold Settlement
  ]
  #v(2cm)
  #text(size: 12pt)[A diploma in Software Engineering]
  #v(3cm)
  #text(size: 12pt)[#emph[Author]]\
  #text(size: 12pt, weight: "bold")[Siryk Mykyta]
  #v(1.5cm)
  #text(size: 11pt)[2026]
]

#pagebreak()

// ---------- Abstract ----------

#heading(level: 1, numbering: none)[Abstract]

We present *Hyperplane-Forge*, a deterministic framework for detecting
synthetically generated faces (deepfakes) by treating an image as a discrete
sample of a continuous physical manifold and asking whether that manifold
admits a smooth energy-minimum settlement. The pipeline first decomposes the
input into a chromatically weighted luminance signal and a multi-scale
difference-of-Gaussians pyramid, then extracts rotationally symmetric
gradients via the Scharr operator and classifies local geometry through the
eigenvalues of the structural tensor. At every edge or corner keypoint we
forge a local linear hyperplane whose slope is derived from the intensity
gradient under a Lambertian reflectance model; these per-keypoint planes
are fused across spatial neighborhoods and frequency scales by a Min-Max
composition that yields the initial manifold $overline(z)_"forged"$. A
global energy functional balances depth fidelity, biharmonic smoothness,
and CNN-trust-weighted gradient consistency; its Euler-Lagrange equation is
solved by a Jacobi fixed-point iteration to produce the settled manifold
$z^*$. Real faces produce smooth $z^*$ and a near-zero residual map; a
deepfake leaves *geometric cracks* — sharp residuals and Laplacian spikes
— wherever the synthetic texture cannot be reconciled with the surrounding
physics. We implement the math kernels in Rust (PyO3, ndarray, rayon) for
the CPU path and mirror them in PyTorch for GPU execution, exposing a
single device-agnostic API to the orchestrator.

#v(1em)
#emph[Keywords:] deepfake detection, physical manifold, Euler-Lagrange,
Lambertian reflectance, structural tensor, Min-Max composition, Rust, PyO3.

#pagebreak()

// ---------- Table of contents ----------

#outline(
  title: [Contents],
  indent: auto,
  depth: 3,
)

#pagebreak()

// ---------- Sections ----------

#include "sections/01-introduction.typ"
#include "sections/02-phase1-luminance.typ"
#include "sections/03-phase2-geometry.typ"
