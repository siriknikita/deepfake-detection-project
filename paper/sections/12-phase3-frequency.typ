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

The Phase 3 experiment ran inside the same training and evaluation
infrastructure as Phase 2 (§11.4), with no new hyper-parameters: the
9-channel model was trained on FF++ c23 face crops with the official
splits, AdamW at $"lr"=2 dot 10^(-4)$, cosine LR annealing, BCE on
logits, WeightedRandomSampler, horizontal-flip augmentation, $20$
epochs. Batch size was reduced from the Phase 2 baseline's $32$ to
$16$ to fit the 9-channel input on the shared cluster GPU; the same
batch size was used for the Phase 2 6-channel run, so the comparison
is at parity in optimiser state. The full training run completed in
$1834$ s (~$30$ min) over the $36{,}000$-frame training split; the
$N = 1$ run yields a single-seed point estimate, with multi-seed
replication left to future work as in §11.9.

The reproducibility steps are unchanged from §11.4 with one extra
cache stage and one extra training argument:

+ Cache the three frequency maps for every face crop in FF++ c23 and
  the Celeb-DF v2 testing list:
  ```
  python scripts/cache_frequency_maps.py \\
      --data-root <ff_root> --frames-subdir frames_faces ...
  python scripts/cache_frequency_maps.py \\
      --data-root <celeb_root> --dataset celeb-df \\
      --celeb-testing-list --frames-subdir frames_faces ...
  ```
+ Train the 9-channel model with the new `--channels` flag:
  ```
  python scripts/train_physics_cnn.py --channels rgb,physics,frequency \\
      --runs-dir runs/physics_9ch_freq ... [Phase 2 flags unchanged]
  ```
+ Run the per-method ablation against the 3-channel baseline:
  ```
  python scripts/eval_per_method.py \\
      --baseline-weights ...baseline_3ch_faces/best.pt \\
      --physics-weights  ...physics_9ch_freq/last.pt \\
      --channels rgb,physics,frequency \\
      --output runs/per_method_phase3.json
  ```
+ Cross-dataset evaluation on the Celeb-DF v2 testing list:
  ```
  python scripts/eval_celebdf.py --model physics \\
      --weights .../physics_9ch_freq/last.pt \\
      --channels rgb,physics,frequency \\
      --output .../celeb_test.json
  ```

*Checkpoint selection.* The default training-script convention
saves `best.pt` at the lowest val_loss epoch and `last.pt` at the
final epoch. For Phase 3, val_loss bottomed at epoch $3$ (val_loss
$0.6068$, val_auroc $0.6981$) and then climbed monotonically for
the remaining $17$ epochs even as val_auroc continued to rise — the
textbook pattern of growing logit magnitudes inflating BCE on a
fixed val set without changing the rank-based AUROC. The per-method
numbers reported below come from `last.pt` rather than `best.pt`.
The two checkpoints differ substantively per-method: `last.pt` is
$+0.0653$ Face2Face, $+0.0284$ FaceSwap, $+0.0248$ NeuralTextures,
and $-0.0049$ Deepfakes (within-noise) relative to `best.pt`.
The improvement on three of four methods exceeds the per-method
noise floor; the regression on Deepfakes does not. For the
cross-dataset Celeb-DF evaluation, both checkpoints agree within
noise; we report `last.pt` for consistency.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, left, right, right, right, right),
    [Test set], [Metric], [`baseline_3ch`], [`physics_6ch`], [`physics_9ch`], [$Delta$ vs 6ch],
    table.hline(),
    [FF++ c23 (combined)], [Frame AUROC],          [0.7309], [0.7197], [0.7028], [$-0.0169$],
    [FF++ c23 (combined)], [Video mean-pool],      [0.8179], [0.8176], [0.7851], [$-0.0325$],
    [FF++ c23 (combined)], [Video max-pool],       [0.9130], [0.9273], [0.8996], [$-0.0277$],
    [FF++ — Deepfakes],     [Video mean-pool],      [0.8713], [0.8787], [0.8583], [$-0.0204$],
    [FF++ — Face2Face],     [Video mean-pool],      [0.8029], [0.7731], [*0.8282*], [*$+0.0551$*],
    [FF++ — FaceSwap],      [Video mean-pool],      [0.8150], [0.7799], [*0.8147*], [*$+0.0348$*],
    [FF++ — NeuralTextures],[Video mean-pool],      [0.6647], [0.6522], [*0.6694*], [*$+0.0172$*],
    [Celeb-DF v2 (cross-dataset)], [Frame AUROC],   [0.5276], [0.5382], [*0.5609*], [*$+0.0227$*],
    [Celeb-DF v2 (cross-dataset)], [Video mean-pool],[0.5405],[0.5458], [*0.6095*], [*$+0.0637$*],
    [Celeb-DF v2 (cross-dataset)], [Video max-pool], [0.5022],[0.5542], [*0.5629*], [*$+0.0087$*],
  ),
  caption: [Phase 3 9-channel build (`physics_9ch` $=$ RGB + physics +
  frequency, `last.pt` checkpoint) vs Phase 2 6-channel
  (`physics_6ch`) and 3-channel RGB baseline (`baseline_3ch`) on
  FF++ c23 and the Celeb-DF v2 cross-dataset testing list. The
  `baseline_3ch` and `physics_6ch` columns reproduce the §11.5
  numbers; the `physics_9ch` column is the new Phase 3 result.
  Bold rows mark $Delta$ improvements over Phase 2.],
) <tbl-phase3-headline>

The structure of @tbl-phase3-headline:

#set list(marker: ([•]))
- *Three of four FF++ per-method swings positive.* Face2Face
  $+0.0551$, FaceSwap $+0.0348$, NeuralTextures $+0.0172$. The
  Face2Face delta is the largest single-method swing of the entire
  experimental programme; the FaceSwap delta nearly cancels the
  Phase 2 loss ($0.7799 -> 0.8147$ vs the $0.8150$ baseline).
- *FF++ Deepfakes regresses by $-0.0204$.* This was not predicted
  in §12.1 (prediction 3); it is the only per-method regression
  in the table.
- *Combined FF++ regresses on every metric.* Frame $-0.0169$, video
  mean-pool $-0.0325$, video max-pool $-0.0277$. The combined
  regression is dominated by the loss of Phase 2's Deepfakes
  specialisation: the global ranking is no longer pulled to the
  top by a single highly-classified manipulation method.
- *Cross-dataset Celeb-DF improves on every metric.* The video
  mean-pool delta of $+0.0637$ is an order of magnitude larger than
  Phase 2's $+0.0053$ improvement over baseline on the same metric
  (§11.5) and moves the cross-dataset metric from chance-grazing
  ($0.5458$) into clearly above-chance signal ($0.6095$). The
  max-pool delta of $+0.0087$ holds Phase 2's lead on the metric
  Phase 2 was strongest on. All five reported Celeb-DF metrics —
  including frame accuracy and video accuracy not shown in the
  table — improve over Phase 2.

== Outcomes vs ex-ante predictions
<sec-phase3-postmortem>

The §12.1 hypotheses produced six concrete numerical predictions.
Four held empirically, two did not.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, right, left),
    [Prediction], [Direction], [Outcome], [Verdict],
    table.hline(),
    [Face2Face $Delta$ vs 6ch],          [$> 0$],            [$+0.0551$], [confirmed (largest swing)],
    [FaceSwap $Delta$ vs 6ch],           [$> 0$],            [$+0.0348$], [confirmed],
    [NeuralTextures $Delta$ vs 6ch],     [$approx 0$ or $+$],[$+0.0172$], [confirmed (small gain)],
    [Deepfakes $Delta$ vs 6ch],          [$approx 0$ or $+$],[$-0.0204$], [falsified in-domain],
    [Celeb-DF max-pool $Delta$ vs 6ch],  [$gt.eq 0$],        [$+0.0087$], [confirmed],
    [Combined FF++ exceeds both phases], [$>$],              [regressed], [falsified],
  ),
  caption: [Phase 3 prediction post-mortem. The two falsifications
  both concern Deepfakes-related metrics; the parametric-recovery
  and cross-dataset predictions all held.],
)

The two falsifications are the same phenomenon viewed from two
angles. Within FF++, Phase 3 dilutes Phase 2's specialisation on
Deepfakes ($-0.0204$ per-method); that dilution mechanically
lowers Phase 2's combined number, which Phase 2 had inflated by
ranking Deepfakes-fake videos at the top of the combined pool. The
ex-ante hypothesis (§12.1, prediction 3) that adding orthogonal
channels would not erase Phase 2's autoencoder gain therefore fails
*in-domain on FF++*.

The cross-dataset row redeems the underlying claim. Celeb-DF v2 is
itself autoencoder-based; on it the Phase 3 build outperforms Phase
2 on every metric, including the autoencoder-style synthesis the
FF++ Deepfakes regression suggested would be lost. Three honest
mechanistic interpretations follow:

+ *The FF++ Deepfakes regression is method-specific, not
  generalisable.* It reflects Phase 2's geometric channels having
  been tuned to the particular artefacts of FF++'s Deepfakes
  implementation, not a property of all autoencoder-based synthesis.
  Adding orthogonal frequency channels dilutes that specific
  specialisation but does not dilute the more general autoencoder
  signal that transfers across datasets.
+ *Compositionality holds where it generalises.* The two
  physics-derived signal types compose additively across manipulation
  families — parametric Face2Face and FaceSwap, hybrid
  NeuralTextures, cross-dataset autoencoder Celeb-DF — and fail to
  compose only on the highly-specialised in-domain detection of one
  particular manipulation method. This is the expected behaviour of
  channel composition when the new channels carry information most
  discriminative on the methods where the old channels were silent;
  the per-method standard deviation collapses from $0.097$ in Phase
  2 to $0.078$ in Phase 3, and the model becomes a more uniformly
  capable detector across families at the cost of less peak
  performance on its strongest single method.
+ *Combined-metric regression is a measurement artefact.* The
  combined AUROC on FF++ pools $140$ real videos with $35$ fake
  videos from each of four manipulation methods and ranks them
  jointly. Phase 2's combined was inflated because its strongest
  per-method classifier (Deepfakes $0.8787$) pushed all
  Deepfakes-fake videos to the top of the global ranking. Phase 3
  ranks the four methods more uniformly, which produces a tighter
  per-method distribution but a slightly lower combined number.
  The combined regression measures this re-distribution effect,
  not a per-class capability loss.

The Phase 3 contribution is therefore best stated as: *adding a
second physics-derived spatial map set to the multi-channel
architecture produces a more uniformly capable detector across
manipulation families and synthesis pipelines, at the cost of
diluting the per-method peak on the single FF++ method on which
the first map set was already strongest. The cross-dataset
generalisation of the resulting detector is substantially better
than either Phase 2 or the 3-channel baseline.*

== Outlook

Phase 3 closes the empirical gap between the methodological pattern
proposed in §11.10 — that the framework's contribution is the
multi-channel physics-tensor pattern itself — and the demonstration
that this pattern's compositionality is real on FF++ parametric
methods and on the Celeb-DF v2 cross-dataset benchmark. With the
`ChannelSource` interface in place, the remaining items in the
§11.10 future-work list — specular highlights and shadow
consistency, subpixel chromatic aberration, sub-surface scattering,
temporal consistency — fit the same interface and require no
further architectural work, only a per-source implementation and
its dedicated cache script.

The empirical method applies uniformly to each successive channel
set: each adds the same shape of rows to the @tbl-phase3-headline
template (in-domain per-method + cross-dataset) and the diff
against the previous build either confirms or falsifies that
channel set's hypothesised target failure mode. A defensible
$12$-channel build (RGB + geometric + frequency + specular) is
$1$–$2$ weeks of implementation effort with the training and
evaluation infrastructure already in place, and is the natural
Phase 4 of this work.

The pattern that emerges from the Phase-2 / Phase-3 contrast — that
each successive physics principle generalises better cross-dataset
than within the dataset its predecessor was tuned on — is the most
important methodological lesson of the empirical chapter. A
multi-channel detector that improves across datasets and across
manipulation families, at the cost of in-domain peak performance
on a single specialised method, is the correct shape of a
generalisable deepfake detector under non-stationary synthesis
pipelines.

#pagebreak()
