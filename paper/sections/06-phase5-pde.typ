= Phase 5: Settlement via the Euler-Lagrange PDE

Phase 5 finds the minimizer $z^*$ of @eq-energy by writing down the
*Euler-Lagrange equation* — the necessary condition for stationarity
of $E(z)$ with respect to admissible variations of $z$ — discretizing
it on the pixel grid, and solving the resulting linear system by
Jacobi fixed-point iteration. The minimizer is the *settled
manifold*; the iteration is the *settlement process*.

== Variational derivative

For an admissible test function $delta z in H^2(Omega)$ with
compact support inside $Omega$, the first variation of $E$ in the
direction $delta z$ is

$ chevron.l E'(z), delta z chevron.r =
  2 integral.double_Omega [thin
    lambda (z - overline(z)_"forged") delta z
    + alpha thin Delta z thin Delta delta z
    + beta thin (W_"cnn"^2 dot.o (nabla I - K nabla z)) dot.c (- K nabla delta z)
  ] dif A. $

Apply integration by parts twice on the biharmonic term and once on
the gradient-consistency term, using the natural boundary conditions
(developed below) so that boundary integrals vanish. We are left with

$ chevron.l E'(z), delta z chevron.r =
  2 integral.double_Omega [thin
    lambda (z - overline(z)_"forged")
    + alpha thin Delta^2 z
    - beta thin op("div") (W_"cnn"^2 dot.o K (nabla I - K nabla z))
  ] delta z thin dif A. $

Stationarity ($chevron.l E'(z), delta z chevron.r = 0$ for all admissible
$delta z$) gives the *Euler-Lagrange PDE*:

$ lambda (z - overline(z)_"forged")
+ alpha thin Delta^2 z
- beta thin op("div") (W_"cnn"^2 dot.o K (nabla I - K nabla z))
= 0
quad "on" Omega. $ <eq-euler-lagrange>

The left-hand side is, by definition, $1/2 dot.c (delta E slash delta z)$,
the *L²-gradient* of the energy. Solving @eq-euler-lagrange means
finding the depth field at which the energy has zero gradient — the
unique stationary point of the convex $E$, hence the unique global
minimum.

== Boundary conditions

Natural boundary conditions for the variational problem on a bounded
domain $Omega$ are read off the boundary integrals dropped above.
Two conditions are needed because the highest derivative in
@eq-euler-lagrange is fourth order:

- *Neumann condition on $z$* — zero normal flux:
  $ partial z slash partial hat(n) = 0 quad "on" partial Omega. $
  This corresponds to the assumption that the depth field continues
  smoothly past the visible image boundary (the face does not have a
  cliff at the edge of the frame).
- *Free-edge condition on $Delta z$* — zero normal flux of the
  Laplacian:
  $ partial (Delta z) slash partial hat(n) = 0 quad "on" partial Omega. $
  This is the elastic-plate "free edge" condition: the boundary is
  not clamped by an external moment.

Both conditions are imposed numerically by *ghost-cell mirroring*:
the depth field is reflected across the image boundary,
$z(-i, j) = z(i, j)$ and $z(H + i - 1, j) = z(H - i - 1, j)$ for the
top and bottom edges and analogously left/right. The Laplacian of a
mirrored field automatically satisfies the second Neumann condition,
so a single mirror handles both.

== Discretization

Place the depth field on a regular pixel grid of spacing $h = 1$ in
both axes. The discrete *5-point Laplacian* at interior pixel
$(i, j)$ is

$ (Delta z)_(i, j) = z_(i+1, j) + z_(i-1, j) + z_(i, j+1) + z_(i, j-1) - 4 z_(i, j). $ <eq-laplacian5>

The *biharmonic operator* is implemented as the iterated Laplacian:
first compute $u = Delta z$ on the full grid (using Neumann mirroring
on the boundary), then $Delta^2 z = Delta u$. This is the same
5-point stencil applied twice, requires one auxiliary buffer, and is
preferred to a fused 13-point biharmonic stencil because it maps
exactly onto a single 2D convolution operator — making the equivalence
test against the GPU-side PyTorch implementation a one-line
expression.

The *divergence* in the third term of @eq-euler-lagrange is
discretized by central differences. Let
$F = (F_x, F_y) = W_"cnn"^2 dot.o K (nabla I - K nabla z)$. Then

$ (op("div") F)_(i, j) = ((F_x)_(i+1, j) - (F_x)_(i-1, j)) / 2
                      + ((F_y)_(i, j+1) - (F_y)_(i, j-1)) / 2. $ <eq-divergence>

Each gradient $nabla z$ and $nabla I$ in the construction of $F$
uses the Scharr operator from Phase 2 — applied to $z$ this is
consistent with the gradient operator already used elsewhere and
keeps numerical errors aligned across all stages.

== Jacobi fixed-point iteration

Discretizing @eq-euler-lagrange yields a sparse linear system

$ M z = b, $

where $M$ is a sparse symmetric positive-definite matrix (positivity
inherited from convexity of $E$, sparsity from the local-stencil
structure of the discretization) and $b$ encodes
$lambda overline(z)_"forged"$ together with the
$op("div") (W_"cnn"^2 K nabla I)$ term that does not depend on $z$.
Splitting $M = D + N$ with $D$ the diagonal of $M$, the *Jacobi
iteration* is

$ z^((n+1))_(i, j) = D_(i, j)^(-1) thin (b_(i, j) - (N z^((n)))_(i, j)). $ <eq-jacobi>

Concretely: at every interior pixel, evaluate the Euler-Lagrange
residual using the *previous* iterate $z^((n))$ for all
neighbor-dependent quantities, then solve the residual for the
*current* pixel under the assumption that all four direct neighbors
are fixed — this gives a closed-form update. The iteration is
*embarrassingly parallel* across pixels: every $z^((n+1))_(i, j)$
depends only on $z^((n))$, so a single rayon-parallelized sweep over
rows produces the next iterate without inter-thread coordination.

=== Convergence criterion

Iteration stops at iteration $n$ when

$ norm(z^((n)) - z^((n-1)))_2 / norm(z^((n)))_2 < tau, $

with default tolerance $tau = 10^(-5)$, or when $n$ reaches a hard
cap $n_max = 500$. In practice convergence is monotone in $E(z^((n)))$
(verified empirically on every run by sampling the energy every ten
iterations) and reaches the tolerance in 100–250 iterations on
$512 times 512$ images.

== Output of Phase 5

The output is the *settled manifold*

$ z^* approx z^((n_"final")), $

a depth field on $Omega$ that minimizes @eq-energy to within
tolerance. Together with the *energy trace* $\(E(z^((n_j)))\)_j$
recorded during iteration, $z^*$ is the only handoff to Phase 6.

#pagebreak()
