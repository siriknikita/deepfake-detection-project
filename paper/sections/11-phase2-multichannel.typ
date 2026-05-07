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

== Physics-map cache

Computing $W_"cnn"$, $z^*$, $R$ on-the-fly inside the training
DataLoader is infeasible: the Jacobi PDE solver runs $1$–$5$ s per
frame, against $30$+ ms target latency for a DataLoader worker. We
therefore precompute the maps once per dataset and store them as
float16 NumPy archives on disk. The cache is laid out as a sibling
tree to the extracted frames:

```
<root>/.../<compression>/frames/<video_id>/<frame>.png
                        physics_<variant>/<video_id>/<frame>.npz
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
be interrupted and continued without bookkeeping. Stored as float16,
the cache is ~$10$ GB for the FF++ c23 heuristic pass at $256 times
256$ with the academic frame cap of $10$ frames per video across all
three official splits, and ~$6$ GB additional for the gtmask pass
restricted to fakes.

Critically, the cache stores *raw float16 arrays*, not coloured PNGs.
Visualisation of the maps (Section 9.3, `viz.panel`) applies viridis or
hot colormaps for human inspection; feeding those to the CNN would
introduce colormap-quantisation artefacts the network would happily
learn instead of the underlying physics.

== Training protocol

All three runs use the same hyperparameters as the Phase-1 baseline so
any AUROC difference reflects the channel composition rather than
optimisation noise:

#table(
  columns: (auto, auto),
  align: (left, left),
  stroke: 0.5pt,
  table.header([*Hyperparameter*], [*Value*]),
  [Optimiser],          [AdamW, $"lr"=10^(-3)$, weight_decay $10^(-4)$],
  [Schedule],           [Cosine annealing, $T_max = 30$ epochs],
  [Batch size],         [$32$],
  [Mixed precision],    [`torch.amp.autocast` on CUDA],
  [Loss],               [BCE on logits, equal class weights via WeightedRandomSampler],
  [Class balancing],    [WeightedRandomSampler with inverse-frequency weights],
  [Image resolution],   [$256 times 256$],
  [Frame caps],         [$10$ train / $10$ val / $10$ test per video],
  [Splits],             [Official FF++ video-disjoint train/val/test],
  [Pretrained backbone], [ImageNet-1K (`EfficientNet_B0_Weights.IMAGENET1K_V1`)],
)

The class-balance sampler corrects for the FF++ $1{:}4$ real-to-fake
ratio that arises from pooling all four manipulation methods into one
fake class. Without it the BCE loss collapses to the majority-class
prediction (the same failure mode visible in the Phase-1 frame-accuracy
of $0.818$ at AUROC $0.378$).

== Experimental sweep

#table(
  columns: (auto, auto, auto, auto),
  align: (left, center, left, left),
  stroke: 0.5pt,
  table.header(
    [*Run*],
    [*Channels*],
    [*$W_"cnn"$ source (train + eval)*],
    [*Question*],
  ),
  [`baseline_3ch`,],
  [$3$],
  [—],
  [Floor — what does an architecturally identical CNN achieve on RGB alone?],
  [`physics_6ch_heuristic`],
  [$6$],
  [Heuristic everywhere],
  [Does the math pipeline contribute signal a CNN finds useful when the trust map is the operationally-deployable variant?],
  [`physics_6ch_gtmask`],
  [$6$],
  [GT mask on fakes; heuristic on reals],
  [Upper bound — what does the combined system achieve under perfect localisation supervision?],
)

All three runs evaluate on the FF++ c23 test split (in-domain) and on
the Celeb-DF v2 testing list (cross-dataset, when local CelebDF copy is
available; cross-dataset eval uses the heuristic cache uniformly since
GT masks do not exist outside FF++). Frame-level AUROC and video-level
AUROC under both mean-pooling and max-pooling are reported in
side-by-side tables identical in format to the Phase-1 results
(Section 10.2).

== Interpretation rubric

The Phase-2 result reads against three rubrics:

+ *Baseline gap*. If `physics_6ch_*` $<$ `baseline_3ch` on FF++ test
  by more than measurement noise ($plus.minus 0.02$ video AUROC at
  $n=300$ test videos), the math pipeline contributes *negative*
  information at the channel level — the CNN learns better from RGB
  alone than from RGB plus distorted spatial maps. This would extend
  the Phase-1 negative result and harden the conclusion that
  Hyperplane-Forge does not contribute discriminative signal at FF++ c23.

+ *Heuristic vs. GT-mask gap*. A large `gtmask` $-$ `heuristic` AUROC
  gap quantifies how much of the missing signal in the operational
  variant is bottlenecked on trust-map localisation. A small gap means
  the heuristic is already good enough that perfect masks would not
  meaningfully improve the detector — which would justify the deployed
  system being heuristic-only.

+ *Cross-dataset transfer*. If the heuristic 6-channel run improves on
  the baseline FF++ → CelebDF transfer (Phase 1: video AUROC $0.500$,
  exact chance), the math pipeline contributes *transferable* signal —
  spatial structure that holds across synthesis pipelines. This is the
  ambitious case, where the framework's hypothesis ("deepfakes leave
  geometric cracks") finds support after the spatial signal is wired
  into the classifier.

The framework's load-bearing claim was that the math pipeline produces
spatial structure deepfakes cannot reproduce. Phase 1 showed the
structure exists but with the wrong sign for a global-pool classifier;
Phase 2 lets a spatial classifier decide what to do with it.

#pagebreak()
