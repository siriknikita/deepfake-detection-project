= Phase 1: Signal Decomposition and Weighted Luminance

Phase 1 conditions the input image so that subsequent geometric stages see
a single-channel, multi-scale signal in which generator-specific noise is
suppressed and surface-relevant structure is preserved. It performs two
operations: a chromatic linear projection that collapses the RGB image
into a luminance scalar with adjustable color weights, and a
difference-of-Gaussians (DoG) pyramid that isolates a discrete set of
spatial-frequency bands.

== Weighted luminance

Let $I: Omega -> [0,1]^3$ denote the input image on the planar pixel
domain $Omega subset RR^2$, with channels $R(x,y), G(x,y), B(x,y) in
[0,1]$. The *weighted luminance* is the linear projection

$ I_w (x, y) = w_R thin R(x,y) + w_G thin G(x,y) + w_B thin B(x,y), $ <eq-luminance>

with non-negative weights $w_R, w_G, w_B in RR_(>=0)$ subject to the
unit-sum constraint

$ w_R + w_G + w_B = 1. $ <eq-weights-sum>

The constraint preserves the dynamic range: $I_w (x,y) in [0,1]$ for any
admissible weight vector. Two reference choices are common in image
processing literature. The ITU-R BT.601 luma weighting

$ (w_R, w_G, w_B)^"BT.601" = (0.299, 0.587, 0.114) $

approximates human photopic sensitivity in the standard-definition
gamut. The BT.709 weighting

$ (w_R, w_G, w_B)^"BT.709" = (0.2126, 0.7152, 0.0722) $

does the same for HD content. Both place the dominant weight on the
green channel, reflecting the higher density of medium-wavelength cones
in the human retina.

For deepfake detection the choice of weights is not perceptually
motivated but *forensically* motivated. GAN- and diffusion-derived
synthesis introduce subtle, model-specific imbalances across the three
color planes — for example, residual checkerboarding from upsampling
operators can be channel-asymmetric, and color-conditional generators
often under-saturate in the blue channel. We therefore treat
$(w_R, w_G, w_B)$ as a *tunable hyperparameter* of the pipeline and
default to BT.601 only when no application-specific calibration is
available. The Phase 6 chapter on impact-map features returns to weight
selection once the residual statistics are defined; the calibration
loop chooses weights to maximize the energy-term separation between
real and forged faces on a held-out validation set.

== Difference-of-Gaussians pyramid

The next operation extracts a stack of band-pass-filtered images from
$I_w$. Let

$ G(x, y; sigma) = 1 / (2 pi sigma^2) thin exp(- (x^2 + y^2) / (2 sigma^2)) $ <eq-gaussian>

denote a two-dimensional isotropic Gaussian kernel of standard deviation
$sigma > 0$, and let $*$ denote convolution on $Omega$ with reflective
(Neumann) boundary handling. The *difference-of-Gaussians* at scale
parameter $k > 1$ is

$ "DoG"_k (x, y) = (I_w * G(dot.c thin ; k sigma)) (x,y) - (I_w * G(dot.c thin ; sigma)) (x,y). $ <eq-dog>

Stacking $K$ such operators with a geometric progression of scales
$sigma, k sigma, k^2 sigma, dots, k^(K-1) sigma$ produces the
*DoG pyramid* $cal(D) = { "DoG"_(k^j) }_(j=0)^(K-1)$. Following Lowe's
convention we take $k = sqrt(2)$ so that adjacent bands cover a half-
octave; with five scales this yields a $4 sqrt(2) approx 5.66 times$
spread between the narrowest and broadest filters.

=== Properties

Three properties of the DoG operator make it the natural input to the
geometric phase that follows.

+ *Zero-mean.* Each $G(dot.c thin ; sigma)$ is unit-norm, so the DC
  component cancels in the subtraction:
  $integral.double_(RR^2) "DoG"_k (x,y) thin dif x thin dif y = 0$.
  Constant illumination biases — overall exposure or gain — therefore
  do not reach the structural-tensor stage.

+ *Band-pass behavior.* The Fourier transform of $"DoG"_k$ is
  $ hat("DoG")_k (xi) = hat(I)_w (xi) thin (exp(-2 pi^2 k^2 sigma^2 |xi|^2) - exp(-2 pi^2 sigma^2 |xi|^2)). $
  The bracketed factor is a band-pass envelope peaked at radial
  frequency $|xi|^* = (1 / (sigma sqrt(2 pi^2 (k^2 - 1)))) sqrt(ln(k^2))$,
  so each scale isolates a narrow frequency annulus.

+ *LoG approximation.* In the limit $k -> 1^+$ the DoG operator
  converges, after rescaling, to the Laplacian of Gaussian:
  $ lim_(k -> 1^+) (k - 1)^(-1) thin "DoG"_k (x, y) = sigma^2 thin Delta (G * I_w) (x, y) $
  where $Delta$ is the spatial Laplacian. This identity is the link
  between Phase 1 and the smoothness term of the energy functional in
  Phase 4: the same second-order operator that defines our band-pass
  filters reappears as the regularizer that controls the curvature of
  the settled manifold.

== Implementation outline

The luminance projection is a fused per-pixel multiply-add, trivially
vectorized. The Gaussian convolutions are implemented as separable 1D
kernels — first a vertical pass, then a horizontal pass — so the cost
per scale is $O(H W r)$ for kernel radius $r approx 3 sigma$ rather
than $O(H W r^2)$. Each DoG band is then a single elementwise
subtraction. The five-scale pyramid for a $512 times 512$ image
costs roughly $5 times 2 r times 512^2 approx 2.6 dot 10^7$
multiply-adds at the default $sigma = 1.0$ — a fraction of a
millisecond on the M4 Pro. Implementation details and the exact
parallelization strategy live in the implementation chapter; for the
purposes of the algorithm specification,
$cal(D) = { "DoG"_(k^j) (I_w) }_(j=0)^(K-1)$ is the only output of
Phase 1.

#pagebreak()
