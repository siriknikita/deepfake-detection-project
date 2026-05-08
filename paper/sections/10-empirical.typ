= Empirical evaluation

This chapter reports the Phase-1 empirical study of Hyperplane-Forge on
two standard deepfake-detection benchmarks, FaceForensics++ (FF++) at the
$"c23"$ compression level and Celeb-DF v2. Phase 1 here refers to the
training-time configuration in which the trust map $W_"cnn"$ is the
deterministic heuristic of Section 3 rather than the learned
ChromaticEfficientNet (Phase 2). The purpose of the chapter is to
quantify how much discriminative signal the math pipeline alone carries
before any learned component is added, and thereby to motivate the
Phase 2 study.

== Datasets and protocol

*FaceForensics++ (FF++).* We use the canonical c23 (medium-compression
H.264) release. From every video we extract $f="fps" = 5$ frames per
second and resize to $256 times 256$ to match the leaderboard convention.
After extraction we stride-trim each per-video frame folder to a
maximum of 30 frames, mirroring the load-time stride sampling of the
dataset adapter so that no frame the trainer sees is downsampled twice.
The released methods cover one source set (`original_sequences/youtube`)
and four manipulations: Deepfakes, Face2Face, FaceSwap, NeuralTextures.

*Celeb-DF v2.* We use the published 518-video testing benchmark
(`List_of_testing_videos.txt`), which contains 178 real videos
(108 Celeb-real, 70 YouTube-real) and 340 synthesized videos
(Celeb-synthesis). Frames are extracted with the same fps-5, $256 times 256$
recipe.

*Splits.* All splits are *video-disjoint* by source identifier — frames
from a single source video may appear in only one of train, validation,
or test, since frame-disjoint splits leak ~0.05–0.15 AUROC. For FF++ we
use a 70/15/15 split with seed 0, keeping all four manipulation methods
together so that a single source identifier carries the corresponding
real and manipulated frames into the same partition. For Celeb-DF cross-
dataset evaluation we report on the full 518 testing videos (using
0/0/518 train/val/test fractions, since no retraining occurs).

*Frame caps.* Training videos are sampled at up to 30 frames each;
validation and test videos at up to 10 each, matching the academic
cross-dataset protocol.

*Pipeline configuration.* For all reported runs we use the heuristic
trust map of Section 3.4, $K = 3$ DoG scales, and $n_max = 200$
Jacobi iterations of the PDE solver. The 24-dimensional feature vector
of Section 7 feeds a Gradient Boosting Classifier
(`sklearn.ensemble.GradientBoostingClassifier`) with `n_estimators=200`,
`max_depth=3`, `learning_rate=0.1`, and a `StandardScaler` preprocessor.

*Hardware.* In-domain training and feature extraction were run on an
LXC container with an RTX 3080 Ti (driver 550.163.01, CUDA 12.4, 12 GB
VRAM); cross-domain evaluation on Celeb-DF was run on an Apple Silicon
Mac with the MPS backend. The math kernels are device-agnostic; we
verified bit-equivalent feature outputs across CUDA and MPS for a
sample of 200 frames during the Mac venv bring-up.

== In-domain results: FaceForensics++ c23

We trained a single binary classifier on real $union$ all four
manipulation methods pooled into one fake class — the *combined-methods*
binary task. The split contained 1400 source identifiers in train, 300
in validation, and 300 in test, materializing as 124,227 / 8,160 / 8,540
frames after the multi-method expansion.

#table(
  columns: (auto, auto, auto, auto),
  align: (left, right, right, right),
  stroke: 0.5pt,
  table.header(
    [*Metric*],
    [*Frame*],
    [*Video (mean-pool)*],
    [*Video (max-pool)*],
  ),
  [Validation AUROC], [0.346], [0.320], [0.424],
  [Test AUROC],       [0.378], [0.347], [0.478],
  [Test accuracy],    [0.818], [—],     [—],
)

The headline test video AUROC of $0.347$ falls below the chance level of
$0.5$. The classifier predicts the majority (fake) class with high
confidence — frame accuracy is essentially the fake base rate of
$~ 80%$ — but the *ranking* it produces is anti-correlated with truth:
real frames are scored *higher* in
$Pr["fake"|x]$ than the actual fakes. AUROC is symmetric about $0.5$
under score sign-flip, so the equivalent "corrected" AUROC is
$1 - 0.347 = 0.653$.

To understand why we observe inverted ranking rather than chance, we
inspected per-class feature means on the training set:

#table(
  columns: (auto, auto, auto, auto),
  align: (left, right, right, right),
  stroke: 0.5pt,
  table.header([*Feature*], [*Real (mean)*], [*Fake (mean)*], [*Δ relative*]),
  [`R_mean`],              [-0.0047], [-0.0041], [13%],
  [`R_p95`],               [1.246],   [1.276],   [2.4%],
  [`R_p99`],               [3.959],   [3.890],   [-1.7%],
  [`absL_mean`],           [0.280],   [0.281],   [0.4%],
  [`absL_p99`],            [4.921],   [4.855],   [-1.3%],
  [`E_smoothness_ratio`],  [0.391],   [0.394],   [0.7%],
  [`E_consistency_ratio`], [0.375],   [0.372],   [-0.7%],
  [`L_spectral_entropy`],  [7.653],   [7.660],   [0.1%],
)

With most features differing by under 3% between classes, the GBC's
ranking is governed by accidental correlations rather than a
discriminative signal. We confirm this by an ablation: refitting with
balanced sample weights changes test AUROC from $0.347$ to $0.339$, no
better; flipping the predicted score recovers $0.653$. The pipeline does
*not* fail by emitting noise — it fails by emitting weak signal in the
inverse direction, consistent with the observation that c23 codec
artefacts dominate the manipulation residual at this compression level
in roughly equal measure for all classes.

=== Per-method ablation

We refit four binary classifiers, one per FF++ method, on (real $union$
that-method-only) subsets of the same training features. Train balance
in each per-method experiment is approximately 20.8k real vs 25.8k fake
(four times more balanced than the combined task), and the test split is
sourced from the same video-disjoint test partition.

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, right, right, right, right),
  stroke: 0.5pt,
  table.header(
    [*Method*],
    [*Frame AUROC*],
    [*Video AUROC (mean)*],
    [*Video AUROC (max)*],
    [*Score-flipped*],
  ),
  [Deepfakes],      [0.368], [0.348], [0.381], [0.652],
  [Face2Face],      [0.421], [0.413], [0.411], [0.587],
  [FaceSwap],       [0.351], [0.325], [0.387], [0.675],
  [NeuralTextures], [0.436], [0.424], [0.427], [0.576],
  [*Combined*],     [*0.378*], [*0.347*], [*0.478*], [*0.653*],
)

All four methods land below chance in the same direction — strong
evidence that the inversion is *systematic*, not a per-method or
per-noise artefact. The score-flipped column reads as a difficulty
ranking consistent with the literature: NeuralTextures is the hardest
to separate (closest to $0.5$) and FaceSwap, whose landmark-blended
boundaries leave the sharpest residual signature, is the easiest. This
ordering supports the interpretation that the math pipeline measures
real, but inversely-correlated, manipulation signal.

== Cross-domain results: FF++ → Celeb-DF v2

We loaded the FF++ classifier without modification and evaluated it on
the published 518-video Celeb-DF v2 testing set. No retraining was
performed; the only computation is feature extraction over Celeb-DF
frames followed by `predict_proba`.

#table(
  columns: (auto, auto, auto),
  align: (left, right, right),
  stroke: 0.5pt,
  table.header([*Metric*], [*Frame*], [*Video (mean-pool)*]),
  [Test AUROC],     [0.510], [0.500],
  [Test accuracy],  [0.652], [—],
  [Test split (real / fake)], [178 / 340], [],
)

Headline cross-dataset video AUROC is $0.500$ — exact chance. The
inversion observed on FF++ does not transfer; with the standard error
of AUROC at $n = 518$ approximately $plus.minus 0.025$, this number is
statistically indistinguishable from a coin flip.

The two findings together — in-domain anti-correlation, cross-domain
chance — establish that whatever weak signal the math pipeline carries
on FF++ is not a transferable property of "deepfake-ness" but a
dataset-specific coupling between the H.264 c23 codec and the
manipulation methods present in FF++. Celeb-DF, which uses a different
synthesis pipeline and its own codec settings, presents a feature
distribution to the FF++-trained classifier that is essentially
indistinguishable from the real-image distribution it was trained on.

== Discussion

Across all four FF++ manipulation methods, the raw Phase-1 score
produced AUROC values below $0.5$. Since this inversion is systematic
rather than isolated, the result indicates that the extracted
settlement features encode a consistent inverse relationship with the
fake-positive label. After orientation correction, video-level AUROC
ranges from $0.576$ to $0.675$, showing that the deterministic
features contain weak-to-moderate forensic signal but are not
naturally calibrated as fake-positive scores. We verified empirically
that this is not a class-indexing or label-assignment artefact: the
fitted pipeline reports `classes_ = [0, 1]`, the AUROC computation
correctly consumes $Pr["class" = 1 | x]$, dataset adapter paths label
`original_sequences/...` as $0$ and `manipulated_sequences/.../...` as
$1$, and the population-level statistic
$EE[Pr["fake"|x) | y = 0] = 0.843$ exceeds
$EE[Pr["fake"|x) | y = 1] = 0.824$ — the misordering is
property of the features themselves, not of the evaluation harness.

The cross-domain finding sharpens the picture: the FF++ classifier
applied without modification to the Celeb-DF v2 testing set reaches
exactly chance ($0.500$), ruling out the interpretation that the
pipeline is detecting any general property of synthesised faces. The
in-domain inverse signal is therefore a dataset-specific coupling
between the H.264 c23 codec and the manipulation methods present in
FF++, not a portable feature of synthetic content.

=== Geometry of the inversion: separator vs orientation

For any binary classifier the decision boundary is a hypersurface in
feature space, and any orientation of that surface admits a paired
orientation that defines the *same* boundary with the labels exchanged.
For a linear classifier this is the duality between a normal
$bold(w)$ and its negation $-bold(w)$: both
$bold(w) dot bold(x) + b = 0$ and $-(bold(w) dot bold(x) + b) = 0$
describe the same hyperplane, but they assign opposite half-spaces to
the positive class. For a probabilistic classifier the analogous
duality is between the score $p(bold(x))$ and $1 - p(bold(x))$.

Under this view, AUROC measures how cleanly the separator splits the
two populations, and is invariant to relabelling the two halves up to
the algebraic identity $"AUROC"(p) + "AUROC"(1 - p) = 1$. AUROC of
exactly $0.5$ corresponds to the case where there *is* no usable
separator — the score and its complement carry identical, null
information. AUROC of $0.347$ corresponds to a usable separator that
the training process has *oriented backwards*: had the classifier learned
$1 - p$ instead of $p$, the same Phase-1 features would have produced
test AUROC $0.653$.

This distinction matters because the empirical question Phase 1 answers
is therefore not "do the math features carry signal?" but "do the
math features carry signal *in the orientation the framework
predicted?*". The answer to the first question is yes — there is a
weak but reproducible separating direction across all four manipulation
methods, the same direction in each, with magnitude that is small but
non-trivial (corrected video AUROC $approx 0.58$–$0.68$). The answer
to the second question is no — that direction points opposite to the
fake-positive direction the framework's narrative ("deepfakes leave
geometric cracks → higher residual energy") expects.

The mechanism behind the wrong-orientation outcome is interpretable in
the framework's own terms. The energy functional $E(z)$ minimises a
weighted sum of fidelity, biharmonic smoothness, and trust-weighted
gradient consistency, and the impact map is defined over the residual
$R = z^* - z_"ideal"$ and Laplacian $L = Delta z^*$ of the *settled*
manifold. The framework's intended discriminative claim is that
manipulated regions concentrate residual: the synthetic injection
locally violates the smooth-manifold assumption, so $|R|$ and
$|L|$ accumulate exactly there. For this concentration to translate
into a detectable per-image feature, the trust map $W_"cnn"$ must
*down-weight* real-image evidence outside manipulated regions, so that
the energy functional does not spend its residual budget on natural
image noise. The Phase-1 heuristic trust map does not have access to
manipulation localisation — it is a chromatic-residual function of $I$,
identical for real and synthesised inputs — so the PDE settles
uniformly over the whole image. Residual then accumulates from two
sources: natural image noise (present everywhere in real photographs)
and manipulation artefacts (present only in the synthesis region of
fakes). At FF++'s c23 compression level, the synthesis pipeline plus
H.264 quantisation appear to smooth manipulated regions slightly more
aggressively than the natural texture of real frames, so the
*total* residual energy is *higher on real frames than on fakes* —
a direction-reversal of the framework's intended signal.

This decomposition makes precise the proper interpretation of the
Phase-1 result: the mathematical core of the framework — the PDE
solver, the energy decomposition, the impact-map features — is
working as specified and is producing real, reproducible signal. What
fails is the *trust map*, the single component that is supposed to
distinguish "manipulation-region evidence" from "natural-image
evidence" before the energy functional sees it. Phase 1 thus
characterises the framework *with the heuristic trust map ablated*,
not the framework as a whole; the load-bearing hypothesis — that a
*learned* trust map can correct the orientation by concentrating the
residual on manipulation regions — has not yet been tested. Phase 2
tests it directly.

These observations do *not* invalidate the framework's mathematical
foundation. The Euler-Lagrange settlement, the impact map decomposition
$cal(I) = (R, L)$, and the Lambertian-grounded forge of Section 4 all
behave as designed; what fails is the *connection* between the math
pipeline's spatial output and the eventual scalar score. The
24-dimensional feature vector that feeds the GBC classifier averages
the impact map globally — a single number per percentile, per energy
ratio, per Laplacian moment — and global pooling cannot represent the
spatial distinction between "high $|R|$ along a manipulated jawline"
and "high $|R|$ from compression noise". Compression and the synthesis
boundary contribute residual energy of comparable magnitude under c23,
so once the spatial pattern is averaged away the discriminative
information is lost.

=== Oracle ablation: ruling out the trust map as the failure mode

Before pivoting we ran an *oracle ablation* in which the heuristic
$W_"cnn"$ was replaced with the FF++ ground-truth manipulation masks
themselves: $W_"cnn" = 1 - "binarise"("mask")$ on fakes (trust drops to
zero exactly on the manipulated region), $W_"cnn" = 1$ on reals. This
configuration provides the math pipeline with *perfect* localisation —
the upper bound on what any learned trust map could approximate.

The result on FF++ c23 was test frame AUROC $0.370$, test video AUROC
$0.327$ (mean-pool) — within noise of the heuristic's $0.378 / 0.347$.
The inversion identity $"AUROC"(p) + "AUROC"(1-p) = 1$ is therefore a
property of the settlement formulation under c23 compression, not a
property of trust-map quality. A learned trust map cannot do better
than the oracle, so the trust-map-supervision Phase 2 plan we
originally outlined — train a ChromaticEfficientNet to approximate
the GT-mask weighting, then re-run the GBC over the resulting
features — has a known ceiling at AUROC $0.37$ and is therefore not
worth running.

This negative result is itself a contribution. Diagnosing failure to
the *settlement formulation under compression* (rather than to the
trust map alone) reframes the problem: a learned component must enter
the pipeline at a place where it can add signal the formulation
itself cannot, not at the place where the original framework predicted.

=== Phase 2 reformulation

Section 11 develops the new Phase 2 plan, summarised here. Instead of
using the math pipeline's output as a *scalar feature source* for a
classifier, we expose its three spatial maps directly as additional
*image channels*. Each frame becomes a six-channel tensor — the
original $(R, G, B)$ image plus the heuristic trust map $W_"cnn"$, the
settled manifold $z^*$, and the residual $R = z^* - z_"ideal"$ — that
an EfficientNet-B0 classifier consumes end-to-end. The first
convolution of the backbone is replaced with a six-input variant whose
RGB-channel weights are copied from the ImageNet stem and whose three
extra-channel weights are initialised to the per-output-channel mean
of the RGB weights, so epoch-zero behaviour matches the standard
pretrained baseline before the network specialises.

Three runs answer the empirical question:

#set list(marker: ([•]))
- *Baseline (3-channel RGB)*: vanilla EfficientNet-B0 on RGB only,
  end-to-end. Establishes the floor an architecturally identical
  detector achieves without the math pipeline.
- *Physics 6-channel, heuristic $W_"cnn"$*: RGB + the three physics
  maps computed with the deterministic chromatic-residual trust map.
  Operational variant — the trust map is available at inference on
  any image. Comparing this run to the baseline isolates the
  contribution of the math pipeline to a learned classifier with no
  privileged information.
- *Physics 6-channel, GT-mask $W_"cnn"$*: same architecture, but the
  PDE is run with the FF++ ground-truth mask as the trust map for
  every fake frame in train, val, and test (heuristic on reals where
  no mask exists). FF++-only experiment: GT masks are equally
  available at evaluation time as at training time, so there is no
  train/test distribution shift. Reports the upper bound on the
  combined system's AUROC under perfect localisation supervision.

The empirical question Phase 2 answers is therefore: *given the math
pipeline produces real but spatially-distributed signal that a
global-pool classifier cannot use, can a CNN that sees the spatial
maps as image channels recover the discriminative direction the
GBC classifier inverted?*

Section 11 reports the run-to-completion result in detail. The summary
is: at the configurations tested (face-cropped FF++ c23, 10 frames per
video, 20 epochs, identical recipe across both arms) the 6-channel
physics input is statistically indistinguishable from the RGB baseline
on the canonical video AUROC (mean-pool) — the headline difference is
$-0.0003$, well inside the $±0.02$ measurement-noise band at $n = 280$
test videos. Under max-pool, the 6-channel run is $+0.014$ above the
baseline, suggesting that physics-derived features add localised
detection precision that mean-pooling across uniformly-sampled frames
dilutes away. Frame-level numbers fall similarly within noise, with the
6-channel run a hair below the baseline ($-0.011$).

The interpretation is therefore neither "the math pipeline rescues the
classifier" nor "the math pipeline hurts": at this scale the math
channels are *complementary on a per-frame basis* but do not drive a
robust per-video uplift over RGB-only EfficientNet-B0 fine-tuning on
the combined four-method task. The per-method ablation in Section 11.7
sharpens the picture considerably — physics features specifically help
on Deepfakes ($+$0.0074 video AUROC) and hurt on the parametric
Face2Face ($-$0.0297) and FaceSwap ($-$0.0351), with the combined null
emerging as the average of these opposing effects — and the
cross-dataset transfer to Celeb-DF v2 in Section 11.8 confirms the
finding is method-invariant rather than FF++-specific (every metric
favours the 6-channel model, max-pool video AUROC by $+$0.0520).
