= Phase 4: The Global Energy Functional

Phase 4 specifies the *variational principle* against which Phase 5
will solve. It defines a functional $E$ on the space of admissible
depth fields whose minimizer $z^*$ is, by construction, the
*physical settled manifold*. The functional combines three terms,
each with a distinct physical interpretation, weighted by three
non-negative scalars $lambda, alpha, beta in RR_(>=0)$.

== The functional

Let $z: Omega -> RR$ be a twice-differentiable depth field. The
*global energy functional* is

$ E(z) = integral.double_Omega [thin lambda thin (z - overline(z)_"forged")^2 thin
  + thin alpha thin (Delta z)^2 thin
  + thin beta thin norm(W_"cnn" dot.o (nabla I - K thin nabla z))^2 thin
  ] dif A. $ <eq-energy>

The three terms encode three different *physical priors* on the
depth field.

=== Data-fidelity term: $lambda (z - overline(z)_"forged")^2$

The first term penalizes deviation from the initial manifold
$overline(z)_"forged"$ produced by Phase 3. It is a quadratic
*L²-attachment* to the hyperplane-forged target, weighted globally by
$lambda > 0$. Two interpretations are useful.

- *Bayesian.* If we model
  $overline(z)_"forged" = z + eta$ with $eta tilde cal(N)(0,
  sigma_d^2 I)$ a noise field, the data term is the negative
  log-likelihood of $overline(z)_"forged"$ given $z$, with
  $lambda = 1 / (2 sigma_d^2)$.
- *Mechanical.* An elastic spring of stiffness $2 lambda$ connects
  every point of the surface $z$ to the corresponding point of
  $overline(z)_"forged"$. Setting $lambda$ controls how loosely or
  tightly the settled surface tracks the forged target.

=== Smoothness term: $alpha (Delta z)^2$

The second term penalizes the squared Laplacian of $z$ — the
*biharmonic regularizer*. It generalizes the membrane-energy
$|nabla z|^2$ used in shape-from-shading toward a *thin-plate
energy*: the Euler-Lagrange minimizer of an isolated
$alpha (Delta z)^2$ term satisfies $Delta^2 z = 0$, the *biharmonic
equation* describing the equilibrium shape of an elastic plate
under prescribed boundary conditions.

The biharmonic regularizer is *strictly stronger* than the membrane
regularizer for our purposes: a piecewise-affine $z$ has zero
membrane energy but non-zero biharmonic energy concentrated on the
piece boundaries. Since $overline(z)_"forged"$ from Phase 3 is
exactly piecewise-affine, the biharmonic term is what provides the
restoring force that settles those boundaries into smooth surfaces.

=== Gradient-consistency term:
$beta norm(W_"cnn" dot.o (nabla I - K thin nabla z))^2$

The third term is the *physically motivated* term and the only
non-trivial coupling between $z$, the image $I$, and the CNN trust
map $W_"cnn"$. Each pixel contributes
$beta thin W_"cnn"(x, y)^2 thin (I_x - K thin z_x)^2 + (I_y - K thin z_y)^2$
to $E$, so the term penalizes pixels where the *image gradient* and
the *predicted depth gradient*, scaled by an *adaptive albedo*
$K(x, y)$, fail to agree.

The factors are:

- $nabla I$ — the all-scales image gradient of $I_w$ (computed once
  via Scharr).
- $K(x, y)$ — the *adaptive albedo coefficient*, an estimate of the
  local Lambertian gain $rho thin |hat(L)|$ that converts surface
  gradients to image gradients. We use the same robust local-median
  estimator as in @eq-gradient-transfer.
- $W_"cnn"(x, y) in [0, 1]$ — the *trust map* produced by the
  Chromatic CNN orchestrator (the Python side, described in the
  implementation chapter). Pixels assigned high trust by the CNN
  contribute strongly to the gradient-consistency term, low-trust
  pixels are effectively excluded.
- $dot.o$ — Hadamard (elementwise) product.

This third term is what *carries the deepfake signal* into the
optimization. Where $W_"cnn"$ flags forgery (low values), the gradient
consistency between $nabla I$ and $K thin nabla z$ contributes little
to the energy and the surface settles to the unconstrained biharmonic
shape — typically smooth and far from the actual image gradients.
Where $W_"cnn"$ is high (trusted real regions), the surface is forced
to track $nabla I / K$ and the settled $z^*$ closely follows the
image. The *spatial mismatch* between the two regimes — the
discontinuity in surface behavior at the boundary of a forged region
— is what the impact map of Phase 6 detects.

== Existence and uniqueness of the minimizer

The functional $E$ is a sum of squares of *linear* operators applied
to $z$ — specifically, the identity, the Laplacian, and the gradient
combined with a multiplication operator. It is therefore *convex* and
*coercive* on the Sobolev space $H^2(Omega)$ provided $alpha > 0$
(coercivity of the biharmonic term controls $|z|_(H^2)$). Standard
calculus of variations on Hilbert spaces then gives existence and
uniqueness of a minimizer

$ z^* = op("argmin", limits: #true)_(z in H^2(Omega)) E(z), $ <eq-minimizer>

subject to natural boundary conditions which we derive in Phase 5.

== The role of the three weights

The triple $(lambda, alpha, beta)$ controls the balance between data
fidelity, smoothness, and physical consistency. Reasonable defaults
calibrated on a small validation set are

$ (lambda, alpha, beta) = (1.0, thin 0.5, thin 5.0), $

with all three on a unit-pixel grid ($h = 1$) and $z$, $I$, $W_"cnn"$
all in $[0, 1]$. The relative magnitudes follow a simple intuition:
$beta$ should be the largest because the gradient-consistency term
carries the actual deepfake signal; $lambda$ should be small enough
that the data term does not pin $z^*$ to the noisy
$overline(z)_"forged"$ but large enough to anchor the global
position; $alpha$ should be large enough to smooth out the
piecewise-affine structure of $overline(z)_"forged"$ but small
enough not to wash out the gradient-consistency signal. The
calibration loop in the implementation chapter describes how to tune
these on a labeled real/forged validation set.

#pagebreak()
