= Phase 3: composing physical principles via frequency-domain channels

Phase 2's per-method ablation revealed a structured signed pattern:
the geometric-chromatic physics tensor adds video-AUROC signal on
autoencoder-based manipulations (FF++ Deepfakes $+0.0074$, Celeb-DF v2
cross-dataset transfer $+0.0053$ mean-pool / $+0.0520$ max-pool) and
removes signal on parametric / graphics-based manipulations (FF++
Face2Face $-0.0297$, FaceSwap $-0.0351$). The interpretation argued in
Section 11.6 is that autoencoder reconstruction blurs micro-geometry
and is therefore visible to the manifold-settlement formulation,
whereas parametric reenactment is geometrically smooth by construction
and leaves the formulation nothing to flag.

This chapter takes that interpretation seriously as a *prediction* and
extends the Phase 2 multi-channel architecture with a second physical
principle that is silent on geometry but active on synthesis residuals
in a different domain. The methodological claim of the framework — that
its contribution is the multi-channel physics-tensor pattern itself,
not the specific three-map composition of $W_"cnn"$, $z^*$ and $R$
(§11.10) — predicts that adding orthogonal physics-derived spatial
maps should recover signal on the methods where the first map set is
silent, *without* being told which method is which.

== Hypothesis

The geometric-chromatic channels of Phase 2 detect deviations from a
locally Lambertian smooth-manifold prior. The frequency channels of
Phase 3 should detect the residue of synthesis pipelines that survive
in the spectral domain: upsampling artefacts from transposed
convolutions, blending boundaries from face-region splice operations,
texture-interpolation residuals from 3DMM-driven reenactment. These
are well-attested in the literature (Frank et al., ICML 2020; Wang et
al., CVPR 2020; Durall et al., CVPR 2020) and are *orthogonal* to the
manifold-residual signal $R$ Phase 2 computes — they live in the
spectrum of the image, not in its differential geometry.

The hypothesis we test by adding three frequency-domain channels is:

#set list(marker: ([•]))
- *Per-method polarity reversal.* Adding frequency channels should
  produce $Delta "AUROC" > 0$ on the FF++ methods where Phase 2's
  physics channels showed $Delta < 0$ (Face2Face, FaceSwap), because
  parametric and graphics-based methods leave spectral signatures even
  when they preserve geometric coherence by construction.
- *Compositionality.* The 9-channel build (RGB + physics + frequency)
  should not erase the Phase 2 win on autoencoder-based methods
  (Deepfakes, Celeb-DF). The two physical principles should compose
  rather than substitute; the conv stem's per-channel learned weights
  give the network the freedom to weight either signal more strongly
  per manipulation method.
- *Cross-dataset robustness.* The Celeb-DF v2 result, which was the
  strongest evidence for Phase 2 (§11.5), should remain or improve
  under the 9-channel build. Frequency artefacts of upsampling are
  among the most cross-dataset-robust signals identified in the GAN-
  detection literature.

The empirical experiment — train the 9-channel build, evaluate it
in-domain on FF++ c23 and cross-dataset on Celeb-DF v2, attribute the
combined number through a per-method ablation — fits inside the same
training and evaluation infrastructure as Phase 2 (Sections 11.3–11.5)
with no new hyper-parameters. Results are reported in Section
@sec-phase3-results below; numbers marked _TBD_ are awaiting compute
allocation on the GPU cluster.

== Three frequency-domain spatial maps

The Phase 3 channels are produced by
`forge_detect.frequency_map.frequency_maps`, a pure-NumPy module with
no SciPy or PyTorch dependency. Each input image
$I in [0, 1]^(H times W times 3)$ is first reduced to its luminance
channel $Y$ under the Rec. 709 weights ($0.2126$ R, $0.7152$ G,
$0.0722$ B) used elsewhere in the pipeline, after which three same-
resolution spatial maps are produced:

#set list(marker: ([•]))
- $E_"DCT" in [0, 1]^(H times W)$ — *per-block AC energy.* For each
  $8 times 8$ block of $Y$, the AC sum
  $sum_(k=1)^(63) |D_k|^2$ over the $63$ non-DC DCT coefficients is
  computed; $log(1 + dot.c)$ compresses the dynamic range; the per-
  block scalar is tiled to $(H, W)$ so each pixel inherits its block's
  energy summary. Per-image min-max maps the result to $[0, 1]$. This
  channel responds to "how much non-flat content lives in this block",
  irrespective of the pattern's specific frequency.
- $H_"DCT" in [0, 1]^(H times W)$ — *per-block high-band ratio.* For
  each $8 times 8$ block, the ratio of upper-half AC energy
  ($"zz" >= 32$) to total AC energy is tiled to $(H, W)$ and per-image
  min-max'd. Sensitive to upsampling / interpolation artefacts that
  push energy into the diagonal high-frequency cells of the block,
  which natural high-frequency content (sharp edges) does not
  uniformly do — natural content typically excites the low-band side
  of the spectrum.
- $F_"FFT" in [0, 1]^(H times W)$ — *full-image FFT log-magnitude.*
  The 2D FFT of $Y$ is taken at the image's full resolution, the
  log-magnitude is computed, `fftshift` centres DC, and per-image
  min-max scales to $[0, 1]$. This channel is the spatially-localised
  counterpart of the radial-spectrum signature Frank et al. and Durall
  et al. used as a global GAN fingerprint; preserving the full $(H, W)$
  layout instead of radial-projecting keeps it consumable by the same
  conv stem as the spatial channels.

The zigzag boundary at position $32$ — the upper half of the AC band —
follows the empirical convention used in JPEG forensics: the diagonal
high-frequency cells $(u + v >= 7)$ for an $8 times 8$ block are the
ones least excited by natural-image content and most diagnostic of
post-quantisation residuals. The boundary is set deliberately strict
so the high-ratio channel reads as "synthesis-likely" rather than as
"any sharp edge".

All three maps are normalised per image, matching the Phase 2 channel
normalisation convention (§11.1). The decision to discard cross-image
scale is the same: the absolute spectrum value is dominated by content
brightness and contrast, neither of which is informative for
manipulation; only the spatial pattern of where energy lands is
informative, and per-image scaling preserves only that.

== Nine-channel input tensor

For an input image $I in [0, 1]^(H times W times 3)$, the Phase 3 input
tensor concatenates the original RGB image, the three Phase 2 physics
channels, and the three Phase 3 frequency channels:

$ X = (I_R, I_G, I_B, W_"cnn", tilde(z)^*, tilde(R), E_"DCT", H_"DCT", F_"FFT") in [0, 1]^(H times W times 9), $ <eq-nine-channel>

producing a $9 times H times W$ tensor whose first three channels
remain the standard ImageNet-normalised RGB inputs, channels 4–6 the
Phase 2 physics tensor, and channels 7–9 the Phase 3 frequency tensor.

The stem-surgery procedure of Section 11.2 generalises: for the new
$32 times 9 times 3 times 3$ stem weight $W^"new"$, channels $0:3$
inherit the ImageNet RGB stem unchanged, and the six new channels are
initialised to the per-output-channel mean of the RGB kernel. This
matches the existing
`build_physics_classifier(in_channels=9, pretrained=True)` factory in
`forge_detect.baseline_cnn`; no architectural change is needed beyond
passing `in_channels=9`. Other hyper-parameters of Phase 2 (AdamW
$"lr" = 2 dot 10^(-4)$, cosine annealing, BCE on logits,
WeightedRandomSampler, horizontal-flip augmentation) carry forward
unchanged.

== Implementation: composable channel sources

The Phase 3 implementation introduces an explicit *channel-source* API
in `forge_detect.datasets` that decouples the on-disk cache locations
of each channel set from the dataset adapters that read them. A
`ChannelSource` is a small dataclass naming the source, declaring how
many channels it contributes, providing the function from image path
to npz-cache path, listing the npz keys to load, and providing a
per-image normalise callable. The `load_channels_concat(image, path,
sources)` helper concatenates RGB with the contribution of every
source in order. Two sources are provided out of the box:

#set list(marker: ([•]))
- `physics_channel_source(variant)` — Phase 2 maps from the
  `physics_<variant>` cache.
- `frequency_channel_source(variant)` — Phase 3 maps from the
  `frequency_<variant>` cache.

The training and evaluation scripts accept a `--channels` flag that
parses a comma-separated spec into a list of sources via
`parse_channel_spec`, e.g.\ `rgb,physics,frequency` for the 9-channel
build or `rgb,physics:gtmask,frequency` for the GT-mask physics +
frequency variant. Adding further channel sets — specular, chromatic
aberration, sub-surface scattering, temporal — is a per-source
implementation effort within the same `ChannelSource` interface; no
changes to the training script, the model factory, or the dataset
adapters are required.

== Caching pipeline

Frequency maps are pre-computed per face crop and stored as float16
`.npz` files under a parallel directory tree:

```
<root>/.../frames_faces/<vid>/<frame>.png
       -> .../frequency_faces_default/<vid>/<frame>.npz
            { dct_block_energy:    (H, W) float16
            , dct_high_ratio:      (H, W) float16
            , fft_radial_logmag:   (H, W) float16 }
```

The caching script `scripts/cache_frequency_maps.py` mirrors the
Phase-2 physics cache: it walks the dataset, atomically writes per-
frame npz files, and is crash-resumable via per-file existence
checks. The per-frame cost is dominated by PIL decode and npz
compression (the DCT + FFT compute is ~$5$ ms per $256^2$ face on a
single CPU core), so a small Python thread pool ($N = 4$) overlapping
decode/encode with the math achieves throughput close to the I/O
ceiling.

== Empirical evaluation
<sec-phase3-results>

The Phase 3 experiment is identical in structure to Phase 2 (§11.4):

+ Cache the three frequency maps for every face crop in FF++ c23 and
  the Celeb-DF v2 testing list:
  ```
  python scripts/cache_frequency_maps.py \\
      --data-root <ff_root> --frames-subdir frames_faces ...
  python scripts/cache_frequency_maps.py \\
      --data-root <celeb_root> --dataset celeb-df \\
      --celeb-testing-list --frames-subdir frames_faces ...
  ```
+ Train the 9-channel model:
  ```
  python scripts/train_physics_cnn.py --channels rgb,physics,frequency \\
      --runs-dir runs/physics_9ch_freq ... [Phase 2 flags unchanged]
  ```
+ Run the per-method ablation against the 3-channel baseline and the
  Phase-2 6-channel model:
  ```
  python scripts/eval_per_method.py \\
      --baseline-weights ...baseline_3ch_faces/best.pt \\
      --physics-weights  ...physics_9ch_freq/best.pt \\
      --channels rgb,physics,frequency \\
      --output runs/per_method_phase3.json
  ```
+ Cross-dataset evaluation on Celeb-DF v2 testing list:
  ```
  python scripts/eval_celebdf.py --model physics \\
      --weights .../physics_9ch_freq/best.pt \\
      --channels rgb,physics,frequency \\
      --output .../celeb_test.json
  ```

The reporting tables below are the same shape as the Phase 2 headline
table. Numbers marked _TBD_ will be filled in after the experiment
runs on the GPU cluster.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, left, right, right, right, right),
    [Test set], [Metric], [`baseline_3ch`], [`physics_6ch`], [`physics_9ch`], [$Delta$ vs 6ch],
    table.hline(),
    [FF++ c23 (combined)], [Frame AUROC],          [0.7309], [0.7197], [_TBD_], [_TBD_],
    [FF++ c23 (combined)], [Video mean-pool],      [0.8179], [0.8176], [_TBD_], [_TBD_],
    [FF++ c23 (combined)], [Video max-pool],       [0.9130], [0.9273], [_TBD_], [_TBD_],
    [FF++ — Deepfakes],     [Video mean-pool],      [0.8713], [0.8787], [_TBD_], [_TBD_],
    [FF++ — Face2Face],     [Video mean-pool],      [0.8029], [0.7731], [_TBD_], [_TBD_],
    [FF++ — FaceSwap],      [Video mean-pool],      [0.8150], [0.7799], [_TBD_], [_TBD_],
    [FF++ — NeuralTextures],[Video mean-pool],      [0.6647], [0.6522], [_TBD_], [_TBD_],
    [Celeb-DF v2 (cross-dataset)], [Frame AUROC],   [0.5276], [0.5382], [_TBD_], [_TBD_],
    [Celeb-DF v2 (cross-dataset)], [Video mean-pool],[0.5405],[0.5458],[_TBD_], [_TBD_],
    [Celeb-DF v2 (cross-dataset)], [Video max-pool], [0.5022],[0.5542],[_TBD_], [_TBD_],
  ),
  caption: [Phase 3 9-channel build vs. Phase 2 6-channel and 3-channel
  baseline on FF++ c23 and Celeb-DF v2. Numbers under `baseline_3ch`
  and `physics_6ch` reproduce the §11.5 results exactly; numbers under
  `physics_9ch` will be filled in after the Phase 3 training and
  evaluation runs complete.],
)

== Predictions

The hypotheses of §12.1 produce concrete numerical predictions the
above table will either confirm or falsify:

+ *Face2Face $Delta$ vs `physics_6ch`*: predicted $> 0$. Phase 2's
  $-0.0297$ is the largest per-method loss; if frequency channels do
  encode parametric-reenactment artefacts, the 9-channel build should
  recover at least some of that.
+ *FaceSwap $Delta$ vs `physics_6ch`*: predicted $> 0$, and likely
  the largest single recovery (Phase 2 lost $-0.0351$ here, the
  largest in the table).
+ *Deepfakes $Delta$ vs `physics_6ch`*: predicted $approx 0$ or
  slightly positive. The autoencoder signal Phase 2 already captures
  is not expected to be erased by adding orthogonal channels, but
  could be additively reinforced.
+ *Celeb-DF max-pool $Delta$ vs `physics_6ch`*: predicted $approx 0$
  or positive. The cross-dataset frequency signature should generalise
  at least as well as the cross-dataset geometric signature did
  (§11.5).
+ *Combined FF++ video AUROC*: predicted to *exceed both `baseline_3ch`
  and `physics_6ch`* on at least the max-pool metric, because the
  null on Phase 2 was the average of opposing per-method effects and
  Phase 3 is hypothesised to flip the negative side.

If the table contradicts these predictions — e.g.\ if Face2Face $Delta$
is negative and Deepfakes $Delta$ is also negative — the
methodological claim that orthogonal physical principles compose
additively in the multi-channel architecture is falsified. Either a
new channel set or a different fusion architecture (e.g.\ per-channel
gating, cross-method routing) would be required.

== Outlook

Phase 3 closes the gap between the methodological pattern proposed in
§11.10 ("the framework's contribution is the multi-channel physics-
tensor pattern itself") and the empirical demonstration of that
pattern's compositionality. With the channel-source API in place, the
remaining items in the §11.10 future-work list — specular highlights /
shadow consistency, subpixel chromatic aberration, sub-surface
scattering, temporal consistency — fit the same interface and require
no further architectural work, only a per-source implementation and
its dedicated cache script. Each successive channel set is an
additional empirical test of the same compositionality claim, with
the same per-method-attribution methodology applied uniformly.

#pagebreak()
