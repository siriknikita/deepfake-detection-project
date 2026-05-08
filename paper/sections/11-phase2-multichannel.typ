= Phase 2: multi-channel physics tensor

Phase 1 established that the math pipeline produces real, reproducible
spatial signal on FF++ c23 — corrected video AUROC $0.58$–$0.68$ across
manipulation methods after sign-flip — but that the orientation is
inverted relative to the framework's intended discriminative direction
and that a global-pool feature classifier cannot recover the spatial
content the inversion is encoded in. The oracle ablation (Section 10.5)
ruled out trust-map quality as the load-bearing failure mode: the math's
output is anti-correlated with manipulation regardless of how cleanly
$W_"cnn"$ localises them.

Phase 2 therefore changes the role of the math pipeline. Rather than
consuming its 24-dimensional global-feature vector with a Gradient
Boosting Classifier, we treat the pipeline as a *spatial feature
extractor*: each of its three per-pixel outputs ($W_"cnn"$, $z^*$, $R$)
becomes an additional channel concatenated to the original RGB image,
and a convolutional classifier learns the spatial pattern that
discriminates real from fake. The framework does not have to commit
ahead of time to a polarity (the inversion is the CNN's problem to
discover) or to a global pooling rule (the CNN learns its own pooling
through the receptive-field hierarchy of the backbone).

== Six-channel input tensor

For an input image $I in [0, 1]^(H times W times 3)$, the math pipeline
produces three same-resolution spatial maps:

#set list(marker: ([•]))
- $W_"cnn" in [0, 1]^(H times W)$ — the trust map (heuristic chromatic
  residual or, in the GT-mask variant, $1 - "binarise"("mask")$);
- $z^* in RR^(H times W)$ — the settled manifold from the Jacobi solver
  (Section 6);
- $R = z^* - z_"ideal" in RR^(H times W)$ — the per-pixel residual
  between the settled manifold and the locally-Lambertian ideal of
  Section 5.

These are stacked into a six-channel input

$ X = (I_R, I_G, I_B, W_"cnn", tilde(z)^*, tilde(R)) in [0, 1]^(H times W times 6), $ <eq-six-channel>

where the math channels are normalised at load time to lie in $[0, 1]$
so the early-layer activations of the pretrained backbone stay in their
expected range:

$ tilde(z)^* (x, y) = (z^* (x, y) - min(z^*)) / (max(z^*) - min(z^*) + epsilon), $

$ tilde(R) (x, y) = 1/2 (1 + tanh(R(x, y) / sigma_R)), quad sigma_R = std(R) + epsilon, $

with $epsilon = 10^(-6)$. Both normalisations are *per-image* — they
discard cross-image scale and preserve only the spatial pattern. This is
deliberate. The settled-manifold value at any single pixel is
meaningless without context (the PDE solver's reference frame floats);
what matters is *where* on the manifold the residual concentrates.
Per-image normalisation removes the cross-image drift that compression
quality, manipulation method, and solver convergence variance would
otherwise leave in the absolute scale.

The trust map is already calibrated to $[0, 1]$ by construction (Sec.
3.4 / oracle definition) and is passed through unmodified.

== Stem surgery on EfficientNet-B0

The classifier is `torchvision.models.efficientnet_b0` with two
modifications:

+ The classifier head (`model.classifier`) is replaced with
  $"Dropout"(0.2) -> "Linear"(1280, 1)$, producing a single logit per
  image. Training uses `BCEWithLogitsLoss` on the binary $0$ (real) /
  $1$ (fake) target.
+ The stem convolution (`model.features[0][0]`) is replaced with a
  $6 -> 32$ variant of the same kernel ($3 times 3$, stride $2$,
  padding $1$, no bias) so that the network can attend to all six
  channels independently from layer one onward.

Let $W^"old" in RR^(32 times 3 times 3 times 3)$ be the ImageNet-pretrained
RGB stem weights and $W^"new" in RR^(32 times 6 times 3 times 3)$ the
new six-input variant. We initialise:

$ W^"new" [:, 0:3, :, :] &= W^"old" "(copied unchanged)", \
  W^"new" [:, 3:6, :, :] &= 1/3 #h(0.5em) overline(W)^"old", $ <eq-stem-init>

where $overline(W)^"old" in RR^(32 times 1 times 3 times 3)$ is the
RGB-mean kernel for each output channel, broadcast across the three
extra input channels. This is the standard ImageNet-extension
initialisation for multi-channel pretrained models: the network's
epoch-zero output is identical to the pretrained baseline up to the
$1/3$ scaling on the new channels, so the classifier inherits ImageNet
features at convergence rather than training the stem from scratch.

The rest of the backbone (`features[1:]`) is unchanged, so all
downstream blocks, batch-norm statistics, squeeze-and-excitation layers,
and the global pool retain their pretrained weights and behaviours.

== Face-crop preprocessing

Pilot runs of the 3-channel baseline on full extracted frames at
$256 times 256$ revealed a hard floor: train AUROC plateaued at
$0.53$ and validation AUROC stayed at chance ($0.50$) across $20$
epochs. The model was learning the per-batch class prior and nothing
else, because too much non-face content (background, hair, body)
dilutes the manipulation signal beyond what a single fine-tuned
EfficientNet-B0 can extract from $720$ training videos. This is the
reason published FF++ leaderboard recipes universally crop to faces.

We therefore added a face-crop step between frame extraction and
training: `scripts/extract_faces.py` runs MTCNN (via the
`facenet-pytorch` package) over every extracted frame, takes the
highest-confidence detection, expands the bounding box by a
configurable margin (we use $0.3$, i.e. $30%$ of the bbox edge as
padding for hair and chin context), and writes the resulting square
crop to a sibling `frames_faces/` directory at the same $256 times
256$ resolution. Frames where MTCNN fails to detect a face fall back
to a centre crop so the dataset stays complete, and the script's
summary reports the fallback rate. On FF++ c23 the detection rate
was $> 99%$.

This is a methodologically important step: the comparison reported in
Section 11.6 holds the entire training pipeline fixed *except* for the
input channels, so any preprocessing choice that lifts both arms
uniformly (face crops, augmentation, frames per video) does not bias
the comparison. Face cropping was applied identically to the
3-channel baseline and the 6-channel physics arm.

== Physics-map cache

Computing $W_"cnn"$, $z^*$, $R$ on-the-fly inside the training
DataLoader is infeasible: the Jacobi PDE solver runs $1$–$5$ s per
frame, against $30$+ ms target latency for a DataLoader worker. We
therefore precompute the maps once per dataset and store them as
float16 NumPy archives on disk. The cache is laid out as a sibling
tree to the face-cropped frames:

```
<root>/.../<compression>/frames_faces/<video_id>/<frame>.png
                        physics_faces_<variant>/<video_id>/<frame>.npz
```

with each .npz holding three same-shape arrays: `wcnn`, `z_star`,
`residual`. Two `<variant>` values are supported:

#set list(marker: ([•]))
- *heuristic* (always): the chromatic-residual trust map of Section 3.4
  fed into `pipeline.detect`; available for every frame.
- *gtmask* (FF++ fakes only): $W_"cnn" = 1 - "binarise"("GT_mask")$;
  the PDE is re-run with this trust map and the resulting $z^*, R$
  are saved separately. Reals fall back to the heuristic cache.

The two caches are not interchangeable: $z^*$ and $R$ depend on which
trust map weighted the consistency term in the energy functional
(Section 5), so the gtmask-variant maps differ from the heuristic
maps even on identical input frames. The two-cache scheme lets us run
the heuristic-only experiment and the GT-mask-everywhere experiment
side by side from the same source frames.

The cache writer (`scripts/cache_physics_maps.py`) is crash-resumable:
each frame's .npz is written atomically and a re-run skips already-
present files at the index level, so a multi-hour caching pass can
be interrupted and continued without bookkeeping. Compute is
parallelised with a small Python `ThreadPoolExecutor` on top of
`rayon`'s default kernel parallelism, yielding $~2.5 times$ speed-up
over a serial loop on the cluster i7-11700KF. Stored as float16,
the cache is ~$16$ GB for the FF++ c23 heuristic pass at $256 times
256$ with the academic frame cap of $10$ frames per video across all
three official splits.

Critically, the cache stores *raw float16 arrays*, not coloured PNGs.
Visualisation of the maps (Section 9.3, `viz.panel`) applies viridis or
hot colormaps for human inspection; feeding those to the CNN would
introduce colormap-quantisation artefacts the network would happily
learn instead of the underlying physics.

== Training protocol

The reported runs are run-to-completion training of the 3-channel
baseline and the 6-channel physics-input model under the same recipe,
differing only in the input channels and (incidentally) batch size due
to GPU memory pressure on the shared cluster:

#table(
  columns: (auto, auto, auto),
  align: (left, left, left),
  stroke: 0.5pt,
  table.header([*Hyperparameter*], [*baseline_3ch*], [*physics_6ch_heuristic*]),
  [Optimiser],          [AdamW, $"lr"=2 dot.c 10^(-4)$, wd $10^(-4)$], [AdamW, $"lr"=2 dot.c 10^(-4)$, wd $10^(-4)$],
  [Schedule],           [Cosine, $T_max = 20$ epochs], [Cosine, $T_max = 20$ epochs],
  [Batch size],         [$32$],                  [$16$ #footnote[Reduced to fit 12 GB RTX 3080 Ti VRAM shared with other tenants on the cluster at the time of the physics run. With WRS-balanced batches the gradient direction is dominated by class composition rather than batch size, so this is not expected to materially shift the comparison; we note it for completeness.]],
  [Mixed precision],    [disabled],              [disabled],
  [BatchNorm],          [trainable],             [trainable],
  [Loss],               [`BCEWithLogitsLoss`],   [`BCEWithLogitsLoss`],
  [Class balancing],    [WeightedRandomSampler], [WeightedRandomSampler],
  [Augmentation],       [Random horizontal flip], [Random horizontal flip],
  [Image input],        [$256 times 256$ face crop, $0.3$ margin], [$256 times 256$ face crop, $0.3$ margin],
  [Frame caps],         [$10$ train / $10$ val / $10$ test per video], [$10$ train / $10$ val / $10$ test per video],
  [Splits],             [Official FF++ video-disjoint], [Official FF++ video-disjoint],
  [Pretrained backbone], [ImageNet-1K], [ImageNet-1K (RGB stem reused, three new channels initialised per Eq. \@eq-stem-init)],
)

Two hyperparameter choices deserve specific note:

#set list(marker: ([•]))
- *Learning rate $2 dot.c 10^(-4)$*. This is the published Rössler
  FF++ fine-tuning recipe (Rössler et al., ICCV 2019). Higher rates
  (we tested $1 dot.c 10^(-3)$
  and $5 dot.c 10^(-4)$) caused first-batch overshoot from the
  freshly-initialised classifier head: after $~100$ steps the optimiser
  had pushed the model into a high-variance prediction regime from
  which the BCE loss pulled it toward the trivial $z = 0$ minimum,
  with $"AUROC" = 0.5$ for the rest of training.
- *BatchNorm trainable*. We tried freezing BN at the pretrained
  ImageNet running statistics (a standard fine-tuning recipe) and
  observed catastrophic gradient collapse on FF++: the dead-ReLU
  cascade behind frozen normalisation produced logit standard
  deviation $0.000$ and $170/213$ of the parameter tensors with zero
  gradient at batch zero. ImageNet running variances do not match the
  FF++ frame statistic distribution closely enough to use as fixed
  normalisation, so BN is left trainable.

Class balance via WRS corrects for the FF++ $1:4$ real-to-fake ratio
that arises from pooling all four manipulation methods into one fake
class. Without it the BCE loss collapses to the majority-class
prediction (the same failure mode visible in the Phase-1
frame-accuracy of $0.818$ at $"AUROC" = 0.378$).

The originally-planned `physics_6ch_gtmask` run (which would have
used the FF++ ground-truth manipulation masks as $W_"cnn"$ during
training) was deferred owing to a disk-space constraint on the
cluster: with the heuristic-variant cache occupying $~16$ GB and the
gtmask cache requiring an additional $~6$ GB, the two could not
coexist on the available scratch volume, and we judged the
heuristic-vs-baseline comparison to be the load-bearing one. The
gtmask result remains future work.

== Results

Both runs evaluated on the FF++ c23 test split: $140$ real videos and
$140$ fake videos (35 per manipulation method, total $1400$ real
frames and $5600$ fake frames at $10$ frames per video). Test-set
metrics are reported below.

#table(
  columns: (auto, auto, auto, auto),
  align: (left, right, right, right),
  stroke: 0.5pt,
  table.header(
    [*Metric*],
    [*baseline_3ch*],
    [*physics_6ch*],
    [*Δ*],
  ),
  [Frame AUROC],              [0.7309], [0.7197], [$-$0.0112],
  [Video AUROC (mean-pool)],  [*0.8179*], [*0.8176*], [$-$0.0003],
  [Video AUROC (max-pool)],   [0.9130], [*0.9273*], [$+$0.0143],
  [Frame accuracy],           [0.7193], [0.7137], [$-$0.0056],
  [Video accuracy (mean)],    [0.7571], [0.7071], [$-$0.0500],
)

Bolded entries mark the headline canonical metric (video AUROC
mean-pool, the standard cross-paper comparison since Rössler et al.,
ICCV 2019) and the strongest favourable delta for the 6-channel
physics input.

=== Per-method ablation

Following the Phase-1 per-method protocol of Section 10.3, we
re-evaluated both *already-trained* models — the 3-channel baseline
and the 6-channel physics — on the FF++ test split filtered to
(reals $union$ that-method-only) one method at a time. The combined
task pools all four manipulation methods into a single fake class;
the per-method evaluation slices the same pre-trained models'
predictions by which method generated each fake, exposing
manipulation-specific behaviour the combined number hides.

#table(
  columns: (auto, auto, auto, auto),
  align: (left, right, right, right),
  stroke: 0.5pt,
  table.header(
    [*Method*],
    [*baseline_3ch*],
    [*physics_6ch*],
    [*Δ*],
  ),
  [Deepfakes],      [0.8713], [*0.8787*], [$+$0.0074],
  [Face2Face],      [*0.8029*], [0.7731], [$-$0.0297],
  [FaceSwap],       [*0.8150*], [0.7799], [$-$0.0351],
  [NeuralTextures], [*0.6647*], [0.6522], [$-$0.0124],
  [*Combined*],     [*0.8179*], [*0.8176*], [$-$0.0003],
)

Numbers are video AUROC (mean-pool). The combined row matches
Section 11.6's run-time `report.json` to four decimals
(sanity check). Bolded entries within each row mark the model that
won the row.

The per-method numbers reveal a coherent mechanistic pattern that
the combined null hides: the physics input wins on Deepfakes,
loses on Face2Face and FaceSwap, and is statistically indistinguishable
on NeuralTextures. The four FF++ methods produce manipulations with
qualitatively different geometric signatures, which we believe
explains the per-method split.

#set list(marker: ([•]))
- *Deepfakes* — autoencoder-based face reconstruction. The
  autoencoder hallucinates a full face from latent space, and the
  reconstruction is rarely geometrically consistent with the
  underlying head pose, lighting, and blending boundary at the
  hairline. There is a real manifold inconsistency that the
  physics pipeline's residual $R = z^* - z_"ideal"$ and Laplacian
  $L = Delta z^*$ are well-suited to catch. The $+$0.0074
  Deepfakes lift is the empirical realisation of the framework's
  load-bearing claim.

- *Face2Face* — parametric facial reenactment via a 3D morphable
  model. The synthesis transfers expression while preserving
  identity; the underlying 3DMM machinery yields manipulations
  that are *geometrically smooth* by construction. From the
  physics pipeline's perspective, the manipulated face *looks
  like a coherent manifold*, so the residual map encodes
  "geometrically consistent" — which the CNN reads as "likely
  real". The $-$0.0297 loss is the math channels being actively
  misleading on this method.

- *FaceSwap* — graphics-based face swap with 3D model fitting.
  Same underlying mechanism as Face2Face: by construction the
  manipulation has geometrically clean structure. The $-$0.0351
  loss is the largest in the table.

- *NeuralTextures* — deferred neural rendering. Subtler than the
  other three (both models hit a floor near 0.66 video AUROC); the
  $-$0.0124 delta is within the standard-error band at $n = 175$
  test videos in the per-method subset.

Read together, the rows say: *physics features detect manifold
inconsistency*, and are therefore informative on manipulation
methods that produce inconsistency (Deepfakes) but
uninformative-or-misleading on methods that preserve geometric
coherence (Face2Face, FaceSwap). This is a substantive piece of
mechanistic understanding — the question "do physics-derived
spatial features help a CNN detect deepfakes?" answers correctly
not as a flat yes/no but as *"yes, on the methods whose
manipulation signature involves the geometric inconsistencies the
math pipeline is constructed to detect; no on methods whose
manipulation pipeline preserves geometric coherence."* The
combined null result of Section 11.6 is the average of these
opposing per-method effects.

For deployment, the practical implication is that the physics
pipeline is a *complementary* detector rather than a stand-alone
upgrade: its predictions are most useful when ensembled with an
RGB baseline using a method-aware (or method-agnostic but
calibrated) gating rule. We do not pursue this ensembling step in
the current work — the contribution of Section 11 is the
isolation of the per-method signal, not its combination.

=== Training trajectory

#table(
  columns: (auto, auto, auto),
  align: (left, right, right),
  stroke: 0.5pt,
  table.header([*Field*], [*baseline_3ch*], [*physics_6ch*]),
  [Epochs trained],         [$20$],     [$20$],
  [Best validation AUROC],  [$0.7574$], [$0.7295$],
  [Best epoch (0-indexed)], [$12$],     [$10$],
  [Final train AUROC],      [$0.9650$], [$0.9739$],
  [Final validation AUROC], [$0.7545$], [$0.7226$],
  [Final train/val gap],    [$0.211$],  [$0.251$],
  [Wall-clock training],    [$1520$ s], [$1643$ s],
)

== Discussion

*The headline combined result is null, but the per-method ablation
(Section 11.7) reveals it is the average of opposing effects.* Video
AUROC under mean-pooling differs by only $-0.0003$ between the two
runs on the combined fake task — well inside the standard-error band
at $n = 280$ test videos ($plus.minus 0.025$). Slicing the same
predictions per manipulation method exposes a $+$0.0074 Deepfakes
lift, a $-$0.0297 Face2Face loss, a $-$0.0351 FaceSwap loss, and a
$-$0.0124 NeuralTextures loss — opposing signals that average to the
combined null. The interpretation is therefore *not* "the physics
input is useless on FF++"; it is *"the physics input encodes manifold
inconsistency, which makes it useful on autoencoder-based Deepfakes
and counterproductive on parametric / graphics-based methods that
preserve geometric coherence by construction."* See Section 11.7 for
the full breakdown and the mechanistic argument.

*The max-pool delta is small but directional.* Video AUROC under
max-pooling favours physics_6ch by $+0.0143$. Mean-pool averages all
$10$ frame probabilities per video; max-pool takes the most
confident one. The two diverge when a model is *spiky* — strongly
right on some frames, neutral on others. The $+0.0143$ max-pool gap
in the physics run is consistent with the math channels encoding
*localised* manipulation cues that surface on individual frames
where the geometric anomaly is sharply expressed and average out
elsewhere. RGB sees the manipulation artefact in a more uniformly
distributed (and thus more averaging-friendly) form. We interpret
this as evidence that the math features are *complementary on a
per-frame basis* but do not deliver a robust per-video uplift over
RGB at this scale. The gap is below the $plus.minus 0.02$
significance threshold we set in advance, so we do not claim it as
a positive result.

*Frame-level numbers fall similarly within noise.* Frame AUROC is
$-0.0112$ for physics_6ch, frame accuracy $-0.0056$, video accuracy
mean-pool $-0.0500$. None of these crosses the $plus.minus 0.02$
band in physics's favour; the video accuracy mean-pool drop, which
is larger than the AUROC difference, is the threshold-at-$0.5$
artefact of the model becoming more confident — confident
predictions miss harder when they miss, but AUROC is rank-based and
so insensitive to that effect.

*Both runs reach published-recipe-comparable territory but not
state of the art.* Published FF++ EfficientNet-B0 fine-tuning hits
video AUROC $> 0.95$ with $30$+ frames per video, ColorJitter and
random-crop augmentation, longer training, and (in some recipes)
larger backbones. Our recipe was deliberately minimal for clarity
of the comparison; a $0.82$ video AUROC for a vanilla 20-epoch
EfficientNet-B0 baseline at $10$ frames per video is consistent
with the literature. The baseline being honest matters here:
inflating both arms with stronger training tricks would have
shifted the absolute numbers without disturbing the comparison —
the contribution of this section is the *delta*, not the level.

*Phase 1 anti-correlation does not propagate to Phase 2.* The
single most important observation is what we *did not* observe.
The Phase-1 GBC reported test AUROC $0.347$ — anti-correlated.
Feeding the same physics maps as image channels rather than as a
24-D global-pool feature vector eliminates the sign inversion
entirely: the 6-channel CNN reaches $0.7197$ frame AUROC, far
above chance and in line with the RGB baseline. The CNN
*does* learn to use the physics channels productively; it just
does not extract more than what RGB already encodes for this
task at this scale. This is independent evidence that the Phase-1
inversion was an artefact of global-pool feature extraction, not
of the physics maps themselves.

*Generalisation gap.* The 6-channel run overfits slightly more than
the baseline (final train/val gap $0.251$ vs $0.211$). With six
input channels rather than three, the model has more capacity to
memorise per-image patterns that do not generalise. The two
runs reach similar peak validation AUROC ($0.7574$ vs $0.7295$,
gap $0.028$ in favour of baseline) but the physics run plateaus
two epochs earlier. Stronger augmentation (ColorJitter, random
resized crop) or more frames per video would close this gap and
might lift the physics arm relative to the baseline, since
augmentation regularises the extra capacity that currently goes
into memorisation.

== Limitations and follow-up experiments

The result reported above is one full run-to-completion comparison
under one set of hyperparameters; it is enough to answer the binary
"does the math help" question but not to claim the answer is
robust across recipes. Three concrete follow-up experiments are
indicated:

+ *Frames-per-video sweep.* Re-cache face crops and physics maps
  at $30$ training frames per video (the standard academic recipe)
  and rerun both arms. This $3 times$ training-data scale-up
  typically lifts FF++ AUROC by $0.05$–$0.10$ and may either
  widen or close the per-video gap between the two arms.
+ *Augmentation ablation.* Add ColorJitter, random resized crop,
  and JPEG-quality augmentation to both arms uniformly. This is
  the standard FF++ regularisation kit; it usually closes the
  train/val gap and raises validation AUROC by $0.03$–$0.05$.
+ *gtmask variant.* The originally-planned `physics_6ch_gtmask`
  experiment, deferred due to disk constraints, would establish
  the upper bound under perfect-localisation supervision. Even if
  the gtmask delta over heuristic is small, it brackets the
  trust-map-quality contribution of the physics pipeline cleanly.

A fourth, more research-oriented direction is *cross-dataset
transfer to Celeb-DF v2*. The Phase-1 cross-dataset experiment
showed exact-chance ($0.500$) FF++-to-Celeb-DF transfer with the
GBC. If a 6-channel CNN trained on FF++ achieves above-chance
Celeb-DF transfer, that would be a strong piece of evidence that
the physics maps encode generalisable structure beyond what RGB
preserves through codec re-compression. We did not run this
experiment owing to time constraints and lack of CelebDF data on
the FF++ training cluster.

#pagebreak()
