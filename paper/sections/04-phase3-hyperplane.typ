= Phase 3: The Hyperplane Forge

Phase 3 is the namesake stage of this work and the *bridge* between the
geometric description produced by Phase 2 and the variational framework
developed in Phase 4. It takes the per-scale gradient fields
$(I_x^k, I_y^k)$ and the per-scale keypoint sets $cal(K)_k$ and
produces a single scalar field $overline(z)_"forged" : Omega -> RR$ —
the *initial depth manifold* against which the energy functional will
later be minimized. Without this stage, $E(z)$ has nothing to be
faithful to, and the Euler-Lagrange settlement is undefined.

The forge proceeds in three substages:

+ Translate the Phase 2 *intensity* gradient at each keypoint into a
  *surface* gradient via a Lambertian reflectance model (Subsection 4.1
  below).
+ Build a *local linear hyperplane* $H_(i, k)$ at every keypoint and
  every DoG scale, parameterized by depth anchor and surface gradient
  (Subsection 4.2).
+ Fuse the family ${H_(i, k)}_(i, k)$ across overlapping spatial
  neighborhoods and across scales by the Min-Max composition that
  produces $overline(z)_"forged"$ (Subsection 4.3).

== Lambertian gradient transfer

We adopt the *Lambertian reflectance model*: at a surface point with
albedo $rho$, surface normal $hat(n)$, and incoming light direction
$hat(L)$, the observed intensity is

$ I = rho thin (hat(n) dot.c hat(L)). $ <eq-lambertian>

Parameterize the surface as the height field $z(x, y)$ over the image
plane; its outward unit normal is

$ hat(n)(x, y) = 1/sqrt(1 + z_x^2 + z_y^2) thin (-z_x, -z_y, 1)^T, $

where $z_x = partial z slash partial x$ and $z_y = partial z slash partial y$.
Take the simplifying assumption that the dominant illuminant lies along
the camera axis, $hat(L) = (0, 0, 1)^T$, so

$ I(x, y) = rho(x, y) / sqrt(1 + z_x^2 + z_y^2). $ <eq-lambertian-camera>

Under the *small-slope regime* $z_x^2 + z_y^2 << 1$ — appropriate for
faces, which subtend a narrow range of normals around the camera-facing
direction — equation @eq-lambertian-camera linearizes to

$ I(x, y) approx rho(x, y) (1 - 1/2 (z_x^2 + z_y^2)). $

Differentiating in $x$ and dropping the $O(|nabla z|^3)$ remainder gives

$ I_x = rho_x - rho thin (z_x z_(x x) + z_y z_(y x)) + O(|nabla z|^3). $

If the albedo is *locally constant* over the keypoint window — the
canonical assumption of differential photometric methods, justified
empirically by the structural tensor's pre-selection of textured
neighborhoods where albedo varies negligibly relative to geometry —
the $rho_x$ term vanishes and we obtain a relation between intensity
gradients and *second* derivatives of $z$. Integrating over the
window once and rearranging, after invoking the small-slope
approximation a second time to drop the cross term $z_y z_(y x)$, we
arrive at the *first-order Lambertian gradient transfer*

$ z_x (x_i, y_i) approx - I_x (x_i, y_i) / (rho(x_i, y_i) + epsilon),
  quad
  z_y (x_i, y_i) approx - I_y (x_i, y_i) / (rho(x_i, y_i) + epsilon), $ <eq-gradient-transfer>

where $epsilon > 0$ is a small denominator regularizer that prevents
amplification in dark pixels and $rho(x_i, y_i)$ is estimated as the
*local median* of $I_w$ within a small window around the keypoint —
the median is robust to the outliers that any single-pixel albedo
estimate would otherwise inherit from sensor noise.

@eq-gradient-transfer is an *approximation*, not an identity. Its
validity is conditional on three assumptions: locally constant albedo,
small surface slope, and dominant axial illumination. These hold to
first order on most face regions but fail on extreme features (deep
nostrils, glasses frames) and under strongly oblique lighting. The
*role* of the Min-Max composition (Subsection 4.3) is precisely to
suppress the contribution of keypoints where the approximation breaks
down — outlier hyperplanes are eroded by the spatial minimum.

== Local hyperplane construction

For every keypoint $i in cal(K)_k$ at every DoG scale $k$, we build a
*local linear depth model* — a hyperplane in the $(x, y, z)$ space —
anchored at $(x_i, y_i)$:

$ H_(i, k) (x, y) = z_(i, k) + z_x|_(i, k) (x - x_i) + z_y|_(i, k) (y - y_i). $ <eq-hyperplane>

The slope coefficients $z_x|_(i, k)$, $z_y|_(i, k)$ come from the
gradient transfer @eq-gradient-transfer applied to the band $"DoG"_k$
at the keypoint. The depth anchor $z_(i, k)$ is taken as

$ z_(i, k) = c_z thin I_w (x_i, y_i), $ <eq-z-anchor>

a *depth-from-intensity* proxy with a global scale factor $c_z > 0$.
The choice of $z_(i, k)$ is calibrated only up to a global affine shift,
because the energy functional (Phase 4) sees only differences
$z - overline(z)_"forged"$ in its data term; an additive constant
$delta in RR$ added to every $z_(i, k)$ produces the same minimizer
$z^* + delta$ shifted by the same constant, and the residual
$R = z^* - z_"ideal"$ is invariant under such shifts when
$z_"ideal"$ is computed by the same procedure.

The *Lambertian reflection model* enters only in @eq-gradient-transfer.
The hyperplane's *value* (i.e., $z_(i, k)$) is set by a non-Lambertian
proxy because depth-from-shading without integration is an ill-posed
problem; what we *can* recover from a single image are *gradients*,
and the hyperplane preserves exactly that information.

== Multi-scale hyperplane field

Stacking $H_(i, k)$ over all keypoints and all scales yields a *family
of hyperplanes*

$ cal(H) = { H_(i, k) }_(k = 0, dots, K-1; i in cal(K)_k). $ <eq-hyperplane-family>

Each pixel $(x, y) in Omega$ is generally *covered* by many members of
$cal(H)$: every keypoint within radius $r$ of $(x, y)$ at every scale
contributes one hyperplane evaluation $H_(i, k)(x, y)$. Let

$ cal(N)_r (x, y) = { i in cal(K)_k : (x_i - x)^2 + (y_i - y)^2 <= r^2 } $

denote the *neighborhood index set* — the set of keypoints (per scale)
within Euclidean distance $r$ of $(x, y)$.

We must reconcile these many overlapping hyperplane estimates into a
single scalar at each pixel. The naive choice — averaging — would
*linearly* combine outliers with inliers and smear sharp geometric
features. Instead we adopt a *non-linear morphological* fusion.

== Min-Max composition

The *Min-Max composition* operates as follows.

*Step 1 (per-scale spatial minimum).* For each scale $k$ independently,
take the minimum hyperplane evaluation across the spatial neighborhood:

$ z_k^"min"(x, y) = min_(i in cal(N)_r (x, y) inter cal(K)_k) H_(i, k)(x, y). $ <eq-min-step>

This is a *morphological erosion* of the hyperplane field on $cal(K)_k$
by a disk of radius $r$. The geometric interpretation is the *lower
envelope* of all admissible local hyperplanes: a pixel is assigned the
*tightest constraint* implied by any nearby keypoint at this scale.

If $cal(N)_r (x, y) inter cal(K)_k = emptyset$ — no keypoint of this
scale lies within radius $r$ — equation @eq-min-step is undefined; we
use the convention $z_k^"min"(x, y) = +infinity$ and resolve undefined
pixels in Step 3.

*Step 2 (max across scales).* Combine the per-scale minima by
taking, at each pixel, the *maximum across scales*:

$ overline(z)_"forged"(x, y) = max_(k in {0, dots, K-1}) z_k^"min"(x, y). $ <eq-max-step>

This is a *morphological dilation* on the discrete scale axis. The
geometric interpretation is *selection of the sharpest agreement*:
each scale $k$ asserts a per-pixel lower bound on the surface;
$overline(z)_"forged"$ is the *strongest such bound* — the scale at
which the local geometry is most decisively constrained.

The combined operation $max_k circle.small min_i$ is the *opening*
on the joint $("space", "scale")$ lattice — a classical morphological
operation that suppresses isolated noise (small-radius, single-scale
outliers are eroded by the inner $min$) while preserving sharp
features that survive across scales (a coherent depth feature is not
diminished by the outer $max$).

*Step 3 (uncovered pixels).* Pixels at which @eq-max-step yields
$+infinity$ — no keypoint in any scale lies within radius $r$ — are
filled in with the per-pixel luminance proxy $c_z thin I_w (x, y)$,
the same expression used as the depth anchor at keypoints. This
*degenerate fallback* contributes no first-order shape information but
produces a finite, non-NaN $overline(z)_"forged"$ everywhere on
$Omega$, which is required for the energy functional to be well-defined.
The fraction of degenerate pixels is small in practice — for
characteristic keypoint densities and $r approx 8$-$12$ pixels, fewer
than 2% of pixels in a typical face image are uncovered.

The full composition is summarized as

$ overline(z)_"forged"(x, y) =
  cases(
    max_k min_(i in cal(N)_r (x, y) inter cal(K)_k) H_(i, k)(x, y) & "if covered",
    c_z thin I_w (x, y) & "otherwise."
  ) $ <eq-min-max>

== Properties of the forged manifold

Three properties of $overline(z)_"forged"$ are critical to the
behavior of the downstream PDE.

+ *Piecewise-linearity.* On any pixel covered by at least one
  hyperplane, $overline(z)_"forged"$ is a maximum of minima of
  affine functions of $(x, y)$, hence itself piecewise-affine.
  Its Laplacian $Delta overline(z)_"forged"$ is therefore zero on
  the interior of each piece and concentrated as a measure on the
  boundaries between pieces — exactly the structure that the
  smoothness term of the energy functional is designed to penalize.

+ *Outlier robustness.* A single misclassified keypoint contributes
  one hyperplane $H_(i^*, k)$ to the inner $min$; if its slope is
  large (a noisy gradient transfer) it produces a *low* value at
  pixels in its neighborhood and is then *erased* by the inner
  $min$ wherever a more reasonable nearby keypoint exists. Outliers
  must be locally consistent across scales to survive the
  composition, which by construction they are not.

+ *Scale-dominance.* Different image features are most informative
  at different scales: fine wrinkles dominate at small $sigma$, the
  gross face shape dominates at large $sigma$. The outer $max$
  selects the dominant scale per pixel, so different parts of the
  forged manifold can come from different bands of $cal(D)$ —
  exactly what would be required to capture multi-scale structure
  in a single scalar field.

== Output of Phase 3

The handoff from Phase 3 to Phase 4 is the single field

$ overline(z)_"forged" : Omega -> RR. $

Together with the gradient field $(I_x, I_y)$ of $I_w$ (a single,
all-scales gradient computed from the unfiltered weighted luminance,
distinct from the per-scale $(I_x^k, I_y^k)$ used internally above),
$overline(z)_"forged"$ is the only carrier of geometric information
into the variational stage.

#pagebreak()
