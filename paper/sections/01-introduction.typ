= Introduction

== Motivation

Generative adversarial networks and diffusion models can now synthesize
human faces that are perceptually indistinguishable from photographs.
Defensive detection has become a moving target: every detector trained on
the artefacts of one generation of synthesis is rapidly defeated by the
next. Most existing detectors phrase the problem in the same way the
generators phrase generation — as a learned classification over pixel
patches — and so inherit the generators' adversarial dynamics. This
diploma takes a different stance.

A photograph of a real face is, in physical terms, a noisy sample of a
smooth surface lit by a continuous illuminant. The *image gradient* is
related to that surface's geometry through reflectance physics; coherent
gradients across spatial scales imply a coherent underlying surface.
A synthetic injection — whether GAN- or diffusion-generated — must locally
reproduce the texture statistics of a face but is not constrained to
preserve this multi-scale geometric coherence. The signature of a deepfake
is therefore not in any one frequency band but in the *failure of physical
consistency between bands and scales*.

== Approach

We model the image as the projection of a continuous depth field
$z: Omega -> RR$ on a planar domain $Omega subset RR^2$. The pipeline
infers an initial estimate $overline(z)_"forged"$ of this field directly
from the image, then asks whether $overline(z)_"forged"$ is close to a
minimizer of a physically motivated energy functional $E(z)$. The
minimizer $z^*$ — the *settled manifold* — is found by solving the
Euler-Lagrange equation of $E(z)$ with a Jacobi fixed-point iteration. A
real face's initial estimate sits in a deep, smooth basin of $E$ and
settles cleanly; a deepfake's initial estimate sits on a steep, rough
landscape and the residual $R = z^* - z_"ideal"$ accumulates the energy
the surface could not shed.

The pipeline runs through six explicit phases:

#set enum(numbering: "1.")
+ *Signal decomposition.* Chromatic weighted luminance $I_w$ and a
  difference-of-Gaussians (DoG) pyramid expose multi-scale structure
  while suppressing the additive Gaussian noise that GAN/diffusion
  outputs typically inherit.
+ *Geometric extraction.* The Scharr operator yields rotationally
  symmetric gradients; the structural tensor classifies each pixel as
  edge, corner, or flat. The edge/corner set provides the keypoints
  $cal(K)$ at which local depth structure is identifiable.
+ *Hyperplane Forge.* At every keypoint $i in cal(K)$ and every DoG
  scale $k$, a local linear hyperplane $H_(i,k)$ is built whose slope
  follows from the Lambertian reflectance model. The per-pixel target
  manifold is the Min-Max composition
  $overline(z)_"forged"(x,y) = max_(k in "scales") min_(i in
  "neigh"(x,y)) H_(i,k)(x,y)$.
+ *Energy functional.* $E(z)$ balances three terms — fidelity to
  $overline(z)_"forged"$, biharmonic smoothness, and CNN-trust-weighted
  gradient consistency between $nabla I$ and $K nabla z$.
+ *Settlement.* The Euler-Lagrange PDE is discretized and solved by
  Jacobi iteration. Each step is parallelized over rows; the energy is
  logged periodically so monotonic decrease can be verified.
+ *Impact map.* The residual $R = z^* - z_"ideal"$ captures global
  *flow breaks*; the Laplacian $L = Delta z^*$ captures local
  *geometric cracks*. Together they form a fixed-length feature vector
  for a downstream binary classifier.

== Contributions

+ A *deterministic, physics-grounded* deepfake detector whose decision
  rule is mechanically derivable from the image rather than learned
  end-to-end. The CNN component (which produces the trust map $W_"cnn"$)
  is auxiliary to, not the substrate of, the decision.
+ A *single-pipeline, two-backend* implementation. The CPU path is a
  PyO3-bound Rust crate using `ndarray` and `rayon` for cache-friendly
  parallel sweeps; the CUDA path is a PyTorch reimplementation of the
  same operators. Both backends emit the identical result schema, and
  cross-backend numerical equivalence is enforced by tests.
+ An explicit, *commit-traceable* construction of every algorithm phase
  — keypoint classifier, hyperplane forge, energy functional, and PDE
  solver — that can be read alongside the present paper, rather than as
  a single end-to-end black-box training script.

== Paper structure

Sections 2 through 7 develop the six pipeline phases mathematically and
in the order in which the algorithm executes them. Section 2 introduces
the chromatic luminance and DoG decomposition. Section 3 covers Scharr
gradients and structural-tensor classification. Section 4 — the namesake
of this work — derives the local hyperplane construction and the Min-Max
composition that produces the initial manifold $overline(z)_"forged"$.
Section 5 specifies the global energy functional, and Section 6 derives
its Euler-Lagrange equation and the Jacobi discretization that settles
it. Section 7 defines the impact map and its statistical features.
Section 8 gives the unified pseudocode. Section 9 describes the
Rust/Python architecture, build system, and verification path; it is
the only chapter that discusses code or deployment, all preceding
chapters being purely mathematical.
