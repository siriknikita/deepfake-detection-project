= Phase 6: The Impact Map

Phase 6 produces the *forensic signature* — the scalar feature vector
on which a downstream binary classifier decides real-versus-forged.
Two scalar fields summarize the settled manifold: the *residual*
$R(x, y)$, which captures *global flow breaks* relative to a smooth
reference; and the *Laplacian* $L(x, y)$, which captures *local
geometric cracks* in the settled surface itself.

== The reference manifold $z_"ideal"$

To define a residual we need a *reference*. We construct it inside
the variational framework rather than from external data:
$z_"ideal"$ is a *low-pass filtered* version of $overline(z)_"forged"$,

$ z_"ideal" (x, y) = (G(dot.c thin ; sigma_"ref") * overline(z)_"forged") (x, y), $ <eq-z-ideal>

with reference scale $sigma_"ref" = 4 sigma$ where $sigma$ is the
finest DoG scale of Phase 1. This $z_"ideal"$ represents *what the
manifold would look like if all high-frequency depth structure were
removed* — a reasonable proxy for the "ideal" smooth face surface.

The choice is deliberately scale-coupled to Phase 1: the same Gaussian
operator that defines our finest band-pass filter defines the
reference. A real face's settled manifold $z^*$ stays close to
$z_"ideal"$ except in regions of legitimate fine-scale detail (eyes,
lips); a deepfake's $z^*$ deviates from $z_"ideal"$ over the full
extent of any forged region, because the gradient-consistency term
of $E(z)$ could not reconcile $nabla I$ with $K nabla z$ there.

== The two impact maps

The *residual map* (the *flow break*) is

$ R(x, y) = z^* (x, y) - z_"ideal" (x, y). $ <eq-residual>

The *Laplacian map* (the *geometric cracks*) is

$ L(x, y) = Delta z^* (x, y), $ <eq-laplacian-map>

discretized by the same 5-point stencil used in Phase 5.
$R$ and $L$ together form the *impact map*; we abbreviate the pair as
$cal(I) = (R, L)$.

The two are *complementary*. $R$ measures the *displacement* of $z^*$
from the smooth reference — large where the surface has settled into a
geometry incompatible with the data. $L$ measures the *curvature* of
$z^*$ — large where the surface has discrete second-order kinks.

== Feature vector

A binary classifier needs a fixed-length numerical vector. We summarize
$cal(I)$ together with the energy decomposition produced by Phase 5
into a 24-dimensional feature vector $bold(f) in RR^24$:

#table(
  columns: 2,
  align: (left, left),
  stroke: 0.5pt,
  table.header([*Feature group (count)*], [*Components*]),
  [Residual statistics (4)],
  [$mu_R, sigma_R, P_(95)(R), P_(99)(R)$],
  [Absolute-Laplacian statistics (4)],
  [$mu_(|L|), sigma_(|L|), P_(95)(|L|), P_(99)(|L|)$],
  [Laplacian morphology (2)],
  [edge-density$(L; tau_L)$, spectral-entropy$(L)$],
  [Energy decomposition (4)],
  [$E^*, E^*_"data", E^*_"smooth", E^*_"cons"$],
  [Energy ratios (2)],
  [$E^*_"smooth" \/ E^*, E^*_"cons" \/ E^*$],
  [Convergence features (3)],
  [$n_"final", Delta E_(0:10), "slope"_(n_"final")(E^((n)))$],
  [Reserved (5)],
  [Held for downstream extensions (gaze, texture, etc.)],
)

The percentile features (rather than mean and max) are robust to
single-pixel outliers and to image-size differences. The energy
*ratios* are *scale-invariant*: an overall change in $|z|$ (e.g.,
from a different choice of $c_z$ in Phase 3) divides every term of
$E^*$ by the same factor and leaves the ratios unchanged. This
isolates the *which-term-dominates* signal — for real faces the
smoothness term should be dominant, for forged regions the
gradient-consistency term should be elevated — from the *absolute
magnitude* signal which is sensitive to global parameter choices.

The convergence features capture a behavior we observed empirically
in calibration runs: a deepfake's energy landscape often *plateaus*
above the global minimum because the gradient-consistency term
cannot be driven to zero in regions of low $W_"cnn"$, and the
solver is forced to terminate at a local trade-off rather than at a
deep minimum. A real face's iteration drops energy quickly in the
first ten iterations and then enters a sub-linear approach to a much
lower plateau.

== Classifier

We train a binary classifier $g: RR^(24) -> [0, 1]$ to map the
feature vector to a deepfake probability:

$ "Pr"("deepfake" | "image") = g(bold(f)). $

The classifier is intentionally a *small* model — we use a Gradient
Boosting classifier (or, alternatively, a regularized logistic
regression) with feature standardization. The decision rule remains
*interpretable*: feature importance from the boosted model tells us
which forensic signal — large $R$, sharp $L$, or anomalous energy
ratios — the classifier finds most discriminative on a given dataset.

== Output of Phase 6

The deepfake probability $"Pr"("deepfake" | "image") in [0, 1]$,
together with the two impact maps $R$ and $L$ for visualization, is
the *output of the entire pipeline*. The visualization layer
(implementation chapter) overlays $|L|$ above its $P_(95)$ threshold
on the input image as the *crack overlay* — the visual artifact that
makes forged regions immediately apparent to a human reviewer.

#pagebreak()
