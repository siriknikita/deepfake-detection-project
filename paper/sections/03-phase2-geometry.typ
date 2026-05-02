= Phase 2: Geometric Extraction and Rotational Symmetry

Phase 2 takes the multi-scale luminance signal of Phase 1 and turns it
into a *geometric description* of the image: a vector field of intensity
gradients $(I_x, I_y)$ at every pixel and a discrete set of *keypoints*
$cal(K)$ at which the local geometry is well-determined. The Phase 3
hyperplane forge (Section 4) consumes only these two outputs.

== Scharr gradient operators

The discrete first-derivative operators in $x$ and $y$ are
implemented as $3 times 3$ convolutions

$ K_x^"Scharr" = 1/32 mat(
  -3,  0, +3;
  -10, 0, +10;
  -3,  0, +3
), quad
K_y^"Scharr" = 1/32 mat(
  -3, -10, -3;
   0,   0,  0;
  +3, +10, +3
), $ <eq-scharr-kernels>

so the gradient at a pixel $(x, y)$ on the input image $I$ (here either
$I_w$ or one of the $"DoG"_k$ bands of Phase 1) is

$ I_x (x,y) = (K_x^"Scharr" * I)(x,y), quad I_y (x,y) = (K_y^"Scharr" * I)(x,y). $ <eq-gradient>

The conventional Sobel kernels use the row pattern $(-1, -2, -1)$ in
place of $(-3, -10, -3)$. Both kernels belong to the *family* of
horizontally antisymmetric, vertically symmetric $3 times 3$
operators

$ K(a, b) = c(a, b) mat(-a, 0, a; -b, 0, b; -a, 0, a), $ <eq-kernel-family>

parameterized by the outer-row coefficient $a > 0$, the center-row
coefficient $b > 0$, and a *calibration constant* $c(a, b)$ chosen so
that the operator is a *consistent* finite-difference approximation to
$partial slash partial x$. To find $c$, demand that $K(a, b)$ applied
to a unit-slope ramp $f(y, x) = x$ returns the true derivative, namely
$1$, at every interior pixel. Substituting the $3 times 3$ patch
$mat((x-1), x, (x+1); (x-1), x, (x+1); (x-1), x, (x+1))$ into the
unnormalized kernel:

$ c(a, b) thin [thin underbrace(-a(x-1) + a(x+1), = 2a) thin
                + thin underbrace(-b(x-1) + b(x+1), = 2b) thin
                + thin underbrace(-a(x-1) + a(x+1), = 2a) thin]
              = c(a, b) thin (4a + 2b) thin stretch(=)^! thin 1, $

so the unique calibration is

$ c(a, b) = 1 / (4 a + 2 b). $ <eq-calibration>

For Scharr $(a, b) = (3, 10)$ this gives $c = 1/32$, the prefactor in
@eq-scharr-kernels above; for Sobel $(a, b) = (1, 2)$ it gives $c = 1/8$.
Note that $c(a, b)$ depends only on the kernel weights, not on the
input — every input gradient of magnitude $g$ produces a discrete
response of exactly $g$ at any interior pixel where the second-order
remainder of the local Taylor expansion vanishes.

@eq-kernel-family and @eq-calibration determine the *response*; what
remains free is the *direction*. Sobel and Scharr disagree not in
calibration but in *rotational variance*: the angular deviation
between the discrete gradient and the continuous gradient $nabla I$
varies with the orientation of an edge.

For an idealized edge with continuous gradient direction $theta$
ranging over $[0, 2 pi)$, the discrete-gradient direction obtained
from $K(a, b)$ deviates from $theta$ by some $epsilon_(a, b)(theta)$.
Scharr's coefficients are the unique positive solution to the
minimax problem

$ min_(a, b > 0) max_(theta) abs(epsilon_(a, b)(theta)), $

namely $(a, b) = (3, 10)$, which minimizes the worst-case angular
error to about $0.1 degree$ — against approximately $5 degree$ for the
Sobel kernel $(a, b) = (1, 2)$. The calibration $c(a, b) = 1 / 32$ for
Scharr is therefore not arbitrary but follows from
@eq-calibration once the rotational-symmetry optimization has selected
the kernel weights. Rotational symmetry is what we need: the structural
tensor (Subsection 3.2) and the Lambertian gradient (Section 4) both
take the gradient *as a vector* and rely on its direction being
correct independent of how the underlying geometry happens to be
oriented in the image plane.

Each kernel sums to zero ($-3 + 0 + 3 = 0$, etc.), so the gradient
operator is *DC-blind*; combined with the zero-mean DoG bands of
Phase 1 this means a constant-illumination shift of the input
propagates as zero through the rest of the pipeline.

== Structural tensor

The gradient field $(I_x, I_y)$ is a *first-order* description of
local geometry. To classify a pixel as edge, corner, or flat we need
a *second-order* description that captures how gradients are
distributed in a small spatial neighborhood. The construction is the
*structural tensor* (also known as the second-moment matrix):

$ J(x, y) = mat(
  chevron.l I_x^2 chevron.r,        chevron.l I_x I_y chevron.r;
  chevron.l I_x I_y chevron.r,  chevron.l I_y^2 chevron.r
), $ <eq-structural-tensor>

where $chevron.l dot.c chevron.r$ denotes a local averaging operator over a
window $W subset Omega$ centered at $(x, y)$:

$ chevron.l f chevron.r (x, y) = sum_((u, v) in W) w(u - x, v - y) f(u, v). $

We use a *box window* of side $2 r_J + 1$ with $w equiv 1 / |W|$ — a
Gaussian window improves the noise-suppression of $J$ marginally but
is slower to compute and gives no advantage at the classifier-threshold
granularity we operate on.

By construction $J(x, y)$ is symmetric and positive semidefinite, so
its two eigenvalues $lambda_1 (x,y) >= lambda_2 (x,y) >= 0$ are real
and admit a closed-form expression. With the abbreviations
$T = "tr"(J) = chevron.l I_x^2 chevron.r + chevron.l I_y^2 chevron.r$ and
$D = det(J) = chevron.l I_x^2 chevron.r chevron.l I_y^2 chevron.r - chevron.l
I_x I_y chevron.r^2$,

$ lambda_(1, 2) (x, y) = T / 2 plus.minus sqrt((T/2)^2 - D). $ <eq-eigenvalues>

The radicand is non-negative because

$ (T/2)^2 - D
  = (chevron.l I_x^2 chevron.r - chevron.l I_y^2 chevron.r)^2 / 4
  + chevron.l I_x I_y chevron.r^2 >= 0, $

with equality if and only if the gradient in $W$ is degenerate (zero
or one-dimensional).

== Edge, corner, and flat classification

The eigenvalue pair $(lambda_1, lambda_2)$ summarizes the shape of
the local gradient distribution.

#table(
  columns: 3,
  align: (left, center, left),
  stroke: 0.5pt,
  table.header([*Class*], [*Eigenvalues*], [*Geometric meaning*]),
  [Flat],   [$lambda_1 approx lambda_2 approx 0$], [No reliable gradient direction; surface is locally constant in luminance.],
  [Edge],   [$lambda_1 >> lambda_2 approx 0$],     [Strong gradient along one direction, none across; consistent with a step in the depth field along a single isophote.],
  [Corner], [$lambda_1 approx lambda_2 >> 0$],     [Strong gradient in two independent directions; consistent with a vertex or corner in the depth field.],
)

The classifier is two thresholds, $tau_"flat"$ and $tau_"edge"$:

$ "class"(x, y) = cases(
  "flat"   & "if" lambda_1 < tau_"flat",
  "edge"   & "if" lambda_1 >= tau_"flat" "and" lambda_2 < tau_"edge",
  "corner" & "if" lambda_2 >= tau_"edge",
) $ <eq-classifier>

with the convention $lambda_1 >= lambda_2$. The thresholds are chosen
on a per-image basis as percentiles of the empirical $lambda_1$ and
$lambda_2$ distributions — typically $tau_"flat" =
"percentile"_30 (lambda_1)$ and $tau_"edge" =
"percentile"_70 (lambda_2)$ — so the classifier adapts to the dynamic
range of each input rather than relying on a global calibration.

The *keypoint set* at scale $k$ is

$ cal(K)_k = {(x, y) in Omega : "class"(x, y) in {"edge", "corner"}}, $ <eq-keypoint-set>

computed from the structural tensor of the band $"DoG"_k$ rather than of
$I_w$ directly. Doing the classification per scale is what makes the
multi-scale Min-Max composition of Phase 3 *informative*: a corner that
is sharp at the finest scale but absent at coarser scales is a
high-frequency feature; one that survives across scales is a robust
geometric anchor.

== Output of Phase 2

For each scale $k in {0, 1, dots, K-1}$, Phase 2 produces

#set list(marker: ([•]))
- the per-pixel intensity gradient $(I_x^k, I_y^k)$, used by Phase 3 to
  derive Lambertian surface gradients, and
- the keypoint set $cal(K)_k$, used by Phase 3 to anchor the local
  hyperplanes $H_(i, k)$.

The pair $({(I_x^k, I_y^k)}_k, {cal(K)_k}_k)$ is the only handoff to
Phase 3.

#pagebreak()
