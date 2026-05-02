= Algorithm

Sections 2 through 7 developed the six phases of Hyperplane-Forge in
the order in which the algorithm executes them. This chapter assembles
those phases into a single procedure with explicit inputs, outputs,
and interfaces between stages, suitable for direct implementation.

== Inputs and outputs

#table(
  columns: 2,
  align: (left, left),
  stroke: 0.5pt,
  table.header([*Symbol*], [*Description*]),
  [$I in [0, 1]^(H times W times 3)$], [RGB image, normalized to unit range.],
  [$W_"cnn" in [0, 1]^(H times W)$],   [Trust map from the Chromatic CNN orchestrator.],
  [$(w_R, w_G, w_B)$], [Chromatic luminance weights (Phase 1).],
  [$(sigma, k, K)$],   [DoG base scale, ratio, and number of scales (Phase 1).],
  [$r_J$],             [Structural-tensor window radius (Phase 2).],
  [$(tau_"flat", tau_"edge")$], [Edge/corner classifier thresholds (Phase 2).],
  [$r$, $c_z$, $epsilon$], [Hyperplane-forge neighborhood radius, depth scale, regularizer (Phase 3).],
  [$(lambda, alpha, beta)$], [Energy-functional weights (Phase 4).],
  [$(tau_"PDE", n_max)$], [Jacobi tolerance and iteration cap (Phase 5).],
  [$g$],               [Trained binary classifier (Phase 6).],
)

The pipeline outputs the deepfake probability $p in [0, 1]$ and, for
visualization and diagnosis, the impact map $cal(I) = (R, L)$ and the
intermediate manifolds $overline(z)_"forged"$ and $z^*$.

== Pseudocode

#align(left)[
#block(stroke: 0.5pt + black, inset: 8pt, radius: 2pt)[
*Algorithm 1.* Hyperplane-Forge: image $-> $ deepfake probability.

#set par(first-line-indent: 0pt)
#set text(font: "DejaVu Sans Mono", size: 9pt)

```
Input:  RGB image I, trust map W_cnn, parameters as listed above.
Output: deepfake probability p, impact map (R, L).

# --------------- Phase 1: Signal Decomposition ---------------
I_w  ← w_R · R + w_G · G + w_B · B                                 # eq. 1
for j ∈ {0, ..., K-1}:
    DoG_j ← I_w * G(σ · k^j) − I_w * G(σ)                          # eq. 4
end for

# --------------- Phase 2: Geometric Extraction ---------------
for j ∈ {0, ..., K-1}:
    (Ix_j, Iy_j) ← Scharr(DoG_j)                                   # eq. 7
    J_j ← box-average of [[Ix_j², Ix_j·Iy_j],
                          [Ix_j·Iy_j, Iy_j²]] over r_J             # eq. 9
    (λ1_j, λ2_j) ← closed-form 2×2 eigenvalues(J_j)                # eq. 11
    K_j ← {pixels classed edge/corner from (λ1_j, λ2_j) and τ}     # eq. 13
end for
(Ix, Iy) ← Scharr(I_w)

# --------------- Phase 3: The Hyperplane Forge ---------------
ρ ← median-filter(I_w, window = 2r+1)
for j ∈ {0, ..., K-1}, for i ∈ K_j:
    zx_i ← −Ix_j(x_i, y_i) / (ρ(x_i, y_i) + ε)                     # eq. 15
    zy_i ← −Iy_j(x_i, y_i) / (ρ(x_i, y_i) + ε)
    z_i  ← c_z · I_w(x_i, y_i)                                     # eq. 17
    H_(i,j)(x, y) ← z_i + zx_i · (x − x_i) + zy_i · (y − y_i)      # eq. 16
end for
for j ∈ {0, ..., K-1}:
    z_j^min(x, y) ← min over i ∈ K_j ∩ N_r(x, y)  of  H_(i,j)(x,y) # eq. 19
end for
z_forged(x, y) ← max over j  of  z_j^min(x, y)                     # eq. 20
fill uncovered pixels (z_forged = +∞) with c_z · I_w               # eq. 21

# --------------- Phase 4 + 5: Energy and Settlement ----------
z^(0) ← z_forged
n ← 0
repeat:
    n ← n + 1
    Δz   ← 5-point Laplacian(z^(n−1))                              # eq. 25
    Δ²z  ← 5-point Laplacian(Δz)
    F_x  ← W_cnn² · K · (Ix − K · ∂_x z^(n−1))
    F_y  ← W_cnn² · K · (Iy − K · ∂_y z^(n−1))
    div_F ← central-difference divergence of (F_x, F_y)            # eq. 26
    residual ← λ (z^(n−1) − z_forged) + α Δ²z − β div_F            # eq. 24
    z^(n) ← z^(n−1) − D⁻¹ · residual                                # eq. 28
    apply Neumann mirroring on the boundary
    if n mod 10 = 0: log E(z^(n))                                  # eq. 22
until ‖z^(n) − z^(n−1)‖₂ / ‖z^(n)‖₂ < τ_PDE  or  n ≥ n_max
z* ← z^(n)

# --------------- Phase 6: Impact and Decision ---------------
z_ideal ← G(σ_ref) * z_forged                                      # eq. 30
R       ← z* − z_ideal                                              # eq. 31
L       ← 5-point Laplacian(z*)                                    # eq. 32
f       ← features(R, L, energy decomposition, convergence)        # 24-D
p       ← g(f)                                                      # binary classifier

return p, (R, L), z_forged, z*
```
]
]

#v(0.4em)

== Phase summary

#table(
  columns: 3,
  align: (center, left, left),
  stroke: 0.5pt,
  table.header([*Phase*], [*Operation*], [*Theoretical purpose*]),
  [1], [Weighted luminance + DoG pyramid],
       [Multi-scale band-pass; suppresses additive noise of generators.],
  [2], [Scharr gradients + structural tensor + classifier],
       [Rotationally symmetric geometry; identifies keypoints with reliable depth.],
  [3.5], [Local hyperplane $H_(i,k)$],
       [Models depth from gradient under Lambertian assumption.],
  [3.6], [Min-Max composition $overline(z)_"forged" = max(min H_(i,k))$],
       [Fuses scales and neighborhoods into the initial manifold.],
  [4], [Global energy functional $E(z)$],
       [Encodes data fidelity, smoothness, and physical consistency as one functional.],
  [5], [Euler-Lagrange PDE + Jacobi solve],
       [Finds the settled state $z^*$ of the manifold (decoherence).],
  [6], [Impact map $R = z^* - z_"ideal"$, $L = Delta z^*$],
       [Detects the flow break (fidelity loss) and geometric cracks.],
)

#pagebreak()
