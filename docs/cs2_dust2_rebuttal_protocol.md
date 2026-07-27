# CounterStrike-1K Dust2 MIRA rebuttal protocol

The original pilot protocol was versioned before inspecting either pilot world-model result. This
document now also records a dated post-pilot amendment, frozen before corrective training or any
inspection of the new confirmatory test clips. It distinguishes exploratory/post-hoc evidence from
the untouched matched-information endpoint.

## Scope and data unit

- Map: Dust2 only.
- Source: CounterStrike-1K 360p WebDataset v12, materialized and verified by
  `scripts/prepare_counterstrike1k.py`.
- Release training support used by the pilot: 39 matches, 835 complete ten-POV rounds, 8,350 POV
  rows, 95.386 aligned POV-hours (the sum over the ten synchronized views inside their common
  temporal intersections).
- Release validation: 3 matches / 54 rounds. Release test consumed by the pilot: 3 disjoint
  matches / 52 rounds.
- No match may cross train, validation, or test. All ten POV rows of a round remain in one split.
- Input: 168x308 RGB, 16 frames at 8 fps. The codec has temporal stride 2, so the model receives
  eight latent frames covering two seconds.
- Action alignment: release row `i` is target-frame aligned and describes the transition into
  source video frame `i`. An emitted observation at source frame `t` is therefore conditioned on
  interval-reduced action rows `t+1` through `t+4`: buttons are OR-reduced and
  `(delta_yaw, delta_pitch)` is summed. No extra normalization is applied before MIRA's unchanged
  native action encoder.
- Ordinary random training and deterministic midpoint validation windows end strictly before each
  POV's released `alive_end_frame`, because the post-death spectator camera is not controlled by
  that player's action stream. Synchronized/shuffled groups apply the same constraint to every
  contributing POV. The preregistered first-death stress window remains event-centered and must
  carry an alive-target mask for generated-motion metrics.

The canonical selection digest, full-manifest digest, source shard list, payload counts, split
statistics, and leakage check are written beside every run. Already materialized payloads may be
reused only if the verifier passes; otherwise the preparation script extracts the selected members
from the derived shard list.

For the v12 360p release used here, the canonical selection SHA-256 is
`5b4733ba910c96221b06923699b6595b27b23ec9659cb1b15032bbf7dceb9cf9` and the full manifest
SHA-256 is `e6d1199595327ccbed7a79760266f1c30fdcf23b717232705c9b11bb6d8707d3`.

### Post-pilot confirmatory split amendment (frozen 2026-07-25 UTC)

The spatial action router described below was selected after looking at the pilot test result.
Therefore the release test is relabeled `pilot_test` and cannot be reused as untouched confirmatory
evidence. `scripts/prepare_cs2_confirmatory_split.py` deterministically hash-ranks only release-train
match identifiers; it never reads video, controls, events, duration, or a model metric. With salt
`cs1k-dust2-spatial-routing-confirmatory-v1`, the first three matches are:

- `0a3129ba726a`
- `798dd34447aa`
- `55065858bcce`

Those matches become the new `test`; the other 36 release-train matches remain `train`; release
validation stays `val`; and release test becomes `pilot_test`. The resulting contract is:

- train: 36 matches / 766 rounds / 7,660 POV rows / 87.090 aligned POV-hours;
- validation: 3 matches / 54 rounds / 540 POV rows / 4.959 aligned POV-hours;
- untouched confirmatory test: 3 matches / 69 rounds / 690 POV rows / 8.296 aligned POV-hours;
- quarantined pilot test: 3 matches / 52 rounds / 520 POV rows.

The semantic selection digest is
`056e60b7bb4435e212f43e6a2e4c2ea0f5c1265978180588ff59689a2fbabb0f`. The generated Parquet
manifest and provenance are verified independently on every training node before launch. Any
change to the salt, selected identifiers, split cardinalities, source-manifest digest, or semantic
digest defines a different experiment.

### Target-frame action-alignment correction (frozen 2026-07-25 UTC)

Before training a model on the untouched confirmatory split, a cross-baseline audit against the
released state transition established that the existing MIRA adapter began each four-row action
interval at the observation row. The release contract is target-frame aligned, so that was one
32-fps row early. The public adapter and its boundary calculations now use rows `t+1` through
`t+4`, with a poison-row regression test proving that row `t` cannot leak into the action for
observation `t`.

The completed 520-POV pilot and the validation-only spatial-routing gate used the earlier interval.
They remain useful historical evidence about the single-model signal and the multiplayer routing
failure/repair, respectively, but neither is relabeled as the corrected 690-POV baseline. The fresh
single-POV baseline, all future confirmatory shared models, and every paper-facing action
evaluation use the corrected adapter. This correction was committed before any new confirmatory
model was trained or evaluated.

The same pre-training audit found that the earlier adapter sampled ordinary windows from the full
rendered round rather than the player-controlled alive interval. The public adapter now binds
`alive_end_frame` from the frozen manifest into every sample and restricts all random/midpoint
plans to it. This is a CS2 data-validity correction: the MIRA model, codec, loss, action encoder,
and normalization path are not otherwise changed by the alive-window fix.

### Fresh single-MIRA endpoint (frozen 2026-07-25 UTC)

The corrected single-POV MIRA baseline is retrained from scratch before any model reads the
untouched 690-POV test split. Baseline naming is deliberately narrow: this is MIRA's native
single-world-model architecture, diagonal flow-matching objective, causal conditioning path, and
action encoder instantiated with the 44.83M-parameter `cs2_small` transformer configuration, not
the upstream 1B configuration. The action encoder's existing mouse clamp/divisor is exposed as a
configuration value and set to 180 degrees for CS2 angular deltas; its embedding and temporal
pooling math are unchanged.

The frozen 123.02M-parameter RAEv2 codec retains MIRA's decoder and loss structure but uses the
public DINOv2 ViT-B/14 backbone and corresponding patch geometry because the manually gated
upstream DINOv3 weights are unavailable on the worker. The result must therefore be labeled
“MIRA architecture, small single-WM, public-backbone baseline,” not a full released-MIRA,
1B/DINOv3, or scaling reproduction.

- A fresh codec is trained on `train` only through optimizer step 18,000, with seed 28, batch size
  4, strict deterministic kernels, no compilation, and validation-only checks every 6,000 steps.
  The endpoint is fixed by step count, not validation or test quality.
- A fresh single-POV world model is trained on `train` only through optimizer step 15,000, with
  seed 28, batch size 10, the native action encoder and normalization, strict deterministic
  kernels, no compilation, and EMA decay 0.999. Fixed validation loss and one fixed-seed rollout
  are recorded every 1,000 steps on `val`; no test row is available to training or stopping.
- The exact confirmatory manifest and provenance SHA-256 values are
  `33abbb623072932431871a612620110c473d4b664c52010e5763c273c6daf10e` and
  `3f6419f9414576c88773874c8009c0814e827186c8782831cc373a27be70c2ef`.
- After the step-15,000 checkpoint is atomically present, evaluate all 69 test rounds / 690 POV
  rows at both midpoint and first-death windows with evaluation seeds 37, 41, and 43. The same
  videos and diffusion RNG are scored under true, next-round same-POV-slot, half-clip
  time-shifted, and zero actions.
- The next-round intervention replaces a receiver round's complete ten-POV action sequences with
  the next different round while leaving receiver videos and POV ordering unchanged. Results are
  retained per round. Confidence intervals resample 69 round clusters and keep the three
  evaluation-seed repeats within each selected cluster; evaluation seeds are not called
  independent model replicates.

This short endpoint is a resource-constrained baseline fixed before its test evaluation. It is not
presented as a converged MIRA scaling result. Diffusion-loss sensitivity is a direct check that the
trained model uses aligned action conditioning, while generated action adherence is evaluated
separately with the common temporal action-recognition and optical-flow suite. Pixel MSE and
Frechet appearance metrics are secondary and cannot by themselves establish action adherence.

### Fresh single-MIRA endpoint result (completed 2026-07-26 UTC)

The fixed endpoint completed without test-based stopping. The codec checkpoint at step 18,000 has
SHA-256 `3c286c59b74cd141e72af69cde1a0a005142d8d2b472c789cdf2a39a140c4b7a`;
the single-WM checkpoint at step 15,000 has SHA-256
`3dbd8f0e43dbe833a5f36370d75f6306c7aa036dfcd3edba767ab138232fa047`.
Training used clean commit `686d3213c3831dea265952b4ead3a79af6f6fb43`. The following results
are for the one preregistered training seed, not independent training replicates:

- Native denoising loss at midpoint was `0.41547` with true actions and `0.45681` with
  same-POV-slot cross-round actions. The paired degradation was `+0.04134` (`+9.95%` relative to
  true), with a 69-round cluster-bootstrap 95% CI of `[0.03909, 0.04364]`; all 69 round means were
  positive. At first death the corresponding values were `0.47360` and `0.51634`, a degradation of
  `+0.04274` (`+9.03%`), CI `[0.04015, 0.04555]`, again positive for all 69 rounds.
- In the separately generated midpoint rollouts, world-view RAFT flow EPE was `0.03524` with true
  actions and `0.04105` with cross-round shuffled actions. Thus true actions reduced EPE by
  `14.2%`; the paired shuffled-minus-true delta was `+0.005812`, CI
  `[0.004953, 0.006698]`, positive in 66/69 rounds. Global-camera flow EPE was `0.02721` versus
  `0.03285`, a `17.2%` reduction and delta `+0.005639`, CI `[0.004744, 0.006576]`,
  positive in 66/69 rounds.
- A frozen 4.15M-parameter temporal action-recoverability probe was selected on real
  training/validation video only (validation macro average precision `0.63344`). On generated
  midpoint video, the preregistered true-target alignment separation was `+0.01470`, with a
  69-round cluster-bootstrap CI of `[0.00659, 0.02893]`. The true-versus-zero separation was
  `+0.02157`, CI `[0.01374, 0.03616]`.

The public CounterStrike-1K addendum at commit
`b8d36e130796292c26cdb862cfae31cdea936e94` also exhaustively applies the same round bootstrap
to every frozen action-probe label, using the already hashed score/label arrays without visual
rescoring. For FIRE, true/shuffled/zero target ARR was `0.156/0.131/0.130` over 85 positive
segments; the true-minus-shuffled estimate was `+0.0246`, CI `[-0.0158, 0.0636]`, so this
per-event row is directional but inconclusive. RELOAD has only three positives and is unsupported.
The complete 14-label addendum has SHA-256
`87a6a47d303e3a611f36236037379bc62216bcc1de925f3aa0621532daed0dc6`. These are exhaustive
post-test diagnostics, not replacements for the preregistered macro endpoint or
multiple-testing-corrected claims.

These three measurements answer different failure modes. Native loss establishes that this MIRA
checkpoint uses the aligned controls; RAFT shows that the benefit reaches generated camera/world
motion; and temporal action recoverability gives action-semantic evidence beyond pixels and flow.
They support action conditioning for this small single-WM Dust2 baseline. They do not establish
perfect pixel prediction, event-level weapon fidelity, a synchronization benefit, convergence, or
the performance of MIRA's full 1B/DINOv3 configuration.

The first standalone generated-rollout export correctly failed before completing a sample because
the operational wrapper enabled deterministic PyTorch algorithms without setting the required
CuBLAS workspace before Python startup. The partial temporary arrays and traceback are preserved
outside the accepted endpoint. Commit `4363a9b1789d6a549f94e69d9f1bdcc24ebae4d4` sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8` in the public exporter before importing PyTorch; it changes
neither training nor the frozen checkpoint. The clean retry produced the accepted archive.

`scripts/audit_cs2_single_confirmatory_endpoint.py` independently rehashes and fails closed over
the manifest, checkpoint, all 24 native evaluation cells and their 69-round sidecars, generated
arrays, RAFT rows, temporal-probe checkpoint and feature archives, action-recoverability artifacts,
and recovery provenance. The completed audit status is `pass`; its report SHA-256 is
`4722361e89c9c5e83d1f60f2dbe78a663a49e3fb44152105ae42b047ec6dc0ae`.

```bash
python scripts/audit_cs2_single_confirmatory_endpoint.py \
  --run-root /runs/20260726_dust2_single_confirmatory_15k_v1 \
  --manifest /data/cs1k-360p/manifest_dust2_confirmatory_spatial_v1.parquet \
  --probe-summary /runs/temporal-arr-cs1k-dust2-v1/probe/summary.json \
  --probe-checkpoint /runs/temporal-arr-cs1k-dust2-v1/probe/temporal_action_probe.pt \
  --output /runs/20260726_dust2_single_confirmatory_15k_v1/post_test_integrity_audit.json
```

## Pilot question: single versus shared MIRA

Question: under an equal GPU wall-clock budget, does a ten-POV shared MIRA baseline trained on
synchronized Dust2 rounds outperform a single-POV MIRA baseline?

Fixed controls:

- one shared, frozen codec excluded from the comparison budget;
- identical DINOv2-B/14 codec, spatial/temporal resolution, action vocabulary, inner transformer
  width/depth, optimizer, seed, and GPU;
- 5.5 timed hours per arm on the same RTX PRO 6000 Blackwell Server Edition;
- ten raw POV rows, 160 video frames, and ten action streams presented per optimizer step
  (single batch 10 versus shared group batch 1 x 10);
- identical loader worker, prefetch, persistence, pinning, and shuffle-buffer settings in both arms;
- final time-limit checkpoint with EMA weights; no best-test-checkpoint selection;
- strict deterministic Torch algorithms and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

Equal wall time is the primary compute match. Because joint spatial attention and independent
batched attention do not have equal FLOP scaling, the report must also disclose optimizer steps,
processed POV frames, throughput, parameter counts, and peak memory. The pilot is evidence about
the released training path, not by itself a causal estimate of synchronization.

## Confirmatory question: synchronization versus matched cross-round grouping

Question: holding the ten-player architecture and information volume fixed, does training on POVs
from the same synchronized round improve prediction relative to ten POVs drawn from different
rounds?

`group_mode=synchronized` and the historical internal name `group_mode=shuffled` use:

- the same ten-player wrapper with `action_routing=spatial`, token/action counts, global batch,
  codec, initialization seed, optimizer, one-node/four-GH200 DDP topology, and fixed
  optimizer-update count;
- the same amended, frozen splits and complete-round eligibility rule;
- different training grouping only.

The `shuffled` code value means **cross-round grouped ten-POV training**: every POV retains its
correct video-action pairing, while the ten members of a group are drawn from different rounds.
It never means shuffled actions and must not be called a "shuffled-action WM" in reporting.

Both trained models are evaluated on synchronized test groups. Evaluating the cross-round-trained
arm on cross-round groups would change the estimand and is prohibited. During training,
`dataset.validation_group_mode=synchronized` also forces both periodic validation and rollout
metrics onto the same synchronized held-out task; only the training loader grouping differs. Run at
least three training seeds; counterbalance arm order across seeds.
`scripts/run_cs2_gh200_sync_control.sh` runs one seed and accepts an explicit arm order, while
`scripts/run_cs2_gh200_sync_control_eval.sh` forces synchronized test grouping. The held-out
launcher also runs paired true, cross-POV-shifted, time-shifted, and zero-action diffusion-loss
interventions for both arms on the same confirmatory midpoint and first-death windows and RNG
seeds. Thus the primary quality comparison is accompanied by direct conditioning-use checks in
ordinary and combat/death context rather than treating visual metrics alone as evidence that
either model uses player actions. Both training arms persist the same deterministic validation
rollout every 1,000 steps for private review, and every node records five-second GPU utilization,
memory, temperature, and power telemetry for the full run.

`CS1K_TRAIN_STEPS` is the primary compute/data match. It gives both arms exactly the same optimizer
updates, raw POV frames, action streams, and model forward/backward operations. `CS1K_ARM_HOURS` is
only a fail-closed safety cap: hitting it makes the run invalid rather than defining an endpoint.
Wall-clock time and loader throughput are recorded as efficiency measurements, not treated as
equal-compute evidence.

### Post-pilot action-routing amendment

The released MIRA multiplayer wrapper adds player identity, projects all ten player action streams,
and then averages the player axis into one global action vector. That vector is broadcast over the
entire ten-POV latent grid. A post-hoc cyclic cross-POV intervention on the completed pilot shared
checkpoint changed projected per-player conditioning by RMS `0.085145`, but the global router
passed only RMS `0.0001268` (mean attenuation `0.001494`; routed cosine `0.999995`). This explains
why the pilot shared model was effectively insensitive to player/action correspondence; it is not a
data-loader mismatch.

The corrective `spatial` mode keeps the parameterization unchanged but broadcasts player `p`'s
projected action only over player `p`'s latent height band before joint spatial attention. A
routing-only counterfactual on that same checkpoint preserved the intervention with mean attenuation
`1.000000`. Both confirmatory arms use this identical router, so the only arm-level treatment remains
synchronized versus shuffled training groups. The old global-router pilot remains reported and is
not retroactively reinterpreted as confirmatory evidence.

### Frozen G7e engineering gate

Before spending GH200 compute, one one-hour synchronized spatial-router model (training seed 28) is
trained on the amended training split using the frozen pilot codec. This is an operational
preflight, not a model-quality comparison. It evaluates only all 54 release-validation rounds with
diffusion seeds 37, 38, and 39. At each 1,000-step validation point, a fixed-seed, fixed-window
rollout MP4 and JSON sidecar are written locally and mirrored to private S3 for live review. The
preflight passes only if:

- spatial routing preserves at least 0.95 of the projected cross-POV action-delta RMS;
- cross-POV-shifted actions raise mean validation loss by at least 0.005 and at least 1% relative to
  true actions; and
- the paired degradation is positive for every one of the three fixed diffusion seeds.

The thresholds are encoded in `scripts/assess_cs2_spatial_routing_preflight.py`. They must not be
relaxed after seeing the run. Failure blocks the GH200 launch and is retained as a reported result.

## Held-out endpoints

The pilot suite evaluates every one of the 52 release-test rounds at its deterministic midpoint
(520 raw POV clips per model and seed). The confirmatory suite instead evaluates every one of the
69 newly frozen test rounds (690 raw POV clips per model and seed).

Primary endpoints:

1. held-out diffusion `loss_total`;
2. Frechet DINO distance versus the codec reconstruction after four unrolled latent frames
   (one generated second).

Secondary endpoints:

- raw Frechet DINO and Inception distances;
- DINO cosine/L2 and latent drift;
- PSNR, LPIPS, and SSIM;
- codec reconstruction floor;
- denoising latency and raw-POV latent throughput;
- train/validation curves at equal optimizer updates and exactly matched processed frames;
- per-arm wall time and loader throughput as disclosed efficiency measurements.

The metric backbone is public `dinov2_vitb14` for all arms. Test rollout seeds are fixed to 37, 38,
and 39 for the pilot and 37 through 41 for the GH200 control. The GH200 evaluator derives the exact
69-round count from the frozen split provenance and rejects a conflicting manual override.
Validation, rollout-metric, and speed phases reseed independently so enabling or skipping one phase
cannot change another. Sample counts must divide batch size exactly; silent truncation is an error.

## Action-conditioning diagnostic

For each final pilot checkpoint, diffusion loss is recomputed on all 520 release-test POV rows
with the same videos, windows, and diffusion RNG under:

- true actions;
- actions cyclically shifted across POV rows;
- actions shifted by half a clip in time;
- zero keyboard/mouse actions.

The paired quantity is `ablated loss - true-action loss`; positive values indicate that the model
uses the correctly aligned actions. Zero actions are an out-of-distribution diagnostic, not the
primary comparison. This diagnostic does not replace event-level evaluation and must not be
described as proof of causal event fidelity.

As a preregistered event-focused secondary diagnostic, repeat that exact intervention on two-second
windows centered at the first in-range `player_death` event in each held-out round. All 52 Dust2
test rounds are eligible. The source-frame start is `death_frame - required_source_frames / 2`,
clamped only at round boundaries, and is identical for all ten POVs and both model arms. Evaluate
all 520 raw POV rows with seeds 37 through 41. This tests whether aligned actions become more
important in immediate combat/death context; it remains a held-out diffusion-loss diagnostic and
must not be reported as generated death-classification accuracy.

For the final GH200 checkpoints, all four action modes are repeated on both midpoint and
first-death windows over all 69 untouched confirmatory rounds / 690 POV rows with seeds 37 through
41. Before any confirmatory model output existed, an annotation-only eligibility check verified
that all 69 rounds contain an in-range `player_death`; it did not read video or model metrics.
These are preregistered secondary endpoints. The validation-only G7e gate and post-hoc pilot
routing diagnostic must remain visibly separated from them.

## Causal future-event representation probe

The model-level synchronization endpoint uses an external probe rather than adding a task head to
MIRA. This preserves the baseline architecture and separates representation quality from
world-model training. For each frozen single, synchronized-trained, and cross-round-grouped
checkpoint:

- load the identical deterministic midpoint clip from every complete train, validation, and
  confirmatory-test round;
- pass only frames/actions `[0, 8)` at 8 fps to MIRA;
- observe the final diffusion-transformer block with a forward hook at clean flow time `tau=1`;
- spatially pool the last causal context latent to one vector per POV;
- reserve frames `[8, 16)` only for one-second post-context event labels.

Future pixels and future actions are never passed to the feature extractor. The primary
model-level comparison gives the synchronized-trained and matched-information cross-round-trained
checkpoints the exact same synchronized ten-POV contexts. Their features have the same dimension
and use identical-capacity heads, optimizer settings, training labels, validation selection, probe
seeds, and paired test windows. The checkpoints differ only in training grouping. The single-MIRA
arm uses one deterministic anchor POV and remains contextual.

The targets are FIRE, RELOAD, damage, weapon switch, player death, item equip, zoom, blind, bomb
plant, defuse, and explosion in the post-context interval. Target support is selected from the
training split only. Report per-target AP/AUROC, macro AP/AUROC, prevalence, excluded low-support
targets, and synchronized-minus-shuffled paired match-cluster bootstrap intervals. The independent
model-level unit remains the world-model training seed; repeated probe initializations quantify
head-fitting sensitivity and must not be reported as world-model replications.
Run `scripts/run_cs2_frozen_event_probe.sh` once inside each audited `seed_*` directory with output
at `seed_*/event_probe`, then run `scripts/summarize_cs2_event_probe_sweep.py` on the sweep root.
The sweep aggregator verifies each synchronized/cross-round checkpoint hash against its child
training audit, averages the three probe-head initializations within a world-model seed, and only
then estimates mean, sample standard deviation, and a training-seed bootstrap interval across at
least three independently trained model pairs.

A separate `input-grouping` DINO probe compares synchronized ten-POV input with an equal-volume
cross-match shuffled input before MIRA training. It is an auxiliary information-content control,
not evidence that a trained world model learned synchronization.

## Reporting and interpretation

- Preserve individual seed JSON files; never report only the best seed.
- Report per-arm mean and sample standard deviation, plus paired arm deltas.
- For the confirmatory result, training-seed variation is the inferential unit. Multiple rollout
  seeds measure sampler variation and are not independent training replicates.
- Report failures and restarts. A run that fails before completing an optimizer step is a preflight
  failure, not a zero-valued result.
- Do not claim a synchronization benefit from the pilot alone. The synchronized-versus-cross-round
  shared architecture is the matched-information test.
- Do not generalize beyond Dust2, the two-second training window, one-second rollout, public
  DINOv2-based codec, or the tested compute range.

## Loader systems gate

The training path uses MIRA's `CounterStrike1KIterable`, not
`cs2_clean.datasets.CS2Dataset`. Both read the same materialized MP4/action payloads and use
TorchCodec, but the classes do not implement the same experimental unit. The `cs2_clean` map-style
loader enumerates overlapping per-POV windows; MIRA samples complete rounds, keeps all ten POVs
contiguous and player-ordered, constructs the cross-round matched-information control, and performs
the preregistered 32-to-8-fps action reduction. Substituting the class is therefore not a
semantics-preserving loader optimization.

The G7e pilot remains fixed at four TorchCodec CPU workers in both arms. It is not restarted or
mutated after observing partial training curves. Before confirmatory GH200 runs, execute
`scripts/bench_cs2_dataloader.py` on the target node type using the exact model-facing batch:
ten POV rows, 16 frames per row, 168x308 RGB, 8 fps, and the released MIRA action reduction.

The benchmark must:

- compare candidate worker settings for both synchronized and shuffled grouping;
- verify byte-identical decoded video, key, and mouse tensors for the first deterministic
  single/synchronized round before timing;
- pin the explicit confirmatory-manifest path and digest rather than rely on filename discovery;
- record the dataset manifest digest, source commit/status, hostname, command, Torch/TorchCodec/CUDA
  versions, hardware, first-batch hashes, and per-case throughput; and
- write an atomic JSON result and fail nonzero on any semantic or timing-case error.

The Slurm publication path runs at least three independent repeats on each of the four allocated
hosts. Duplicate paths, duplicate payloads, reused completion timestamps, wrong-host evidence, a
non-GH200 GPU, or fewer than four distinct hosts fail closed. One common worker count is selected
by maximizing the lower of synchronized and shuffled mean input throughput over every node/repeat;
an exact score tie chooses fewer workers. The chosen queue configuration is then revalidated
against only the current node's evidence before that node enters training.

Changing only the number of CPU workers or queue settings is allowed after this gate because it
does not change decoded samples. A GPU/DALI backend additionally requires a preregistered decoded
pixel tolerance and a model-quality control because different H.264 decoders are not assumed to be
pixel-identical. The selected loader settings are frozen identically for all confirmatory arms and
training seeds, preserved in resolved configs and node launcher provenance, and audited separately
from model quality.

For an isolated benchmark node that does not hold the full training materialization,
`scripts/stage_cs2_loader_benchmark_val.sh` downloads the three validation matches from the pinned
Hugging Face dataset revision, verifies the manifest and six shard SHA-256 digests, extracts only
the `val`/Dust2 payloads, and writes the exact selection/download provenance. This avoids reading
or copying from a live training volume.

## Reproduction entry points

- Data selection/materialization: `scripts/prepare_counterstrike1k.py`
- Untouched confirmatory split: `scripts/prepare_cs2_confirmatory_split.py`
- Isolated loader-benchmark staging: `scripts/stage_cs2_loader_benchmark_val.sh`
- Exact-contract data-loader benchmark: `scripts/bench_cs2_dataloader.py`
- Allocated-node GH200 loader preflight: `scripts/run_cs2_gh200_loader_preflight.sh`
- Slurm-native one-node/four-GH200 seed orchestration: `scripts/run_cs2_gh200_slurm_seed.sh`
- G7e pilot: `scripts/run_cs2_rebuttal_pipeline.sh`
- Paired pilot evaluation: `scripts/run_cs2_rebuttal_eval.sh`
- Action loss diagnostic: `scripts/run_cs2_action_loss_ablation.sh`
- Spatial-router representation diagnostic: `scripts/diagnose_cs2_multi_action_routing.py`
- Validation-only spatial-router preflight: `scripts/run_cs2_spatial_routing_preflight.sh`
- Frozen preflight gate: `scripts/assess_cs2_spatial_routing_preflight.py`
- First-death-centered action diagnostic: `scripts/run_cs2_death_action_ablation.sh`
- Unattended paired-evaluation guard: `scripts/watch_cs2_rebuttal_eval.sh`
- Unattended midpoint-action guard: `scripts/watch_cs2_action_loss_ablation.sh`
- Unattended event-diagnostic guard: `scripts/watch_cs2_death_action_ablation.sh`
- Unattended final certification guard: `scripts/watch_cs2_rebuttal_audit.sh`
- GH200 matched control: `scripts/run_cs2_gh200_sync_control.sh`
- GH200 held-out evaluation: `scripts/run_cs2_gh200_sync_control_eval.sh`
- Frozen MIRA context features: `scripts/extract_cs2_future_event_features.py`
- Causal frozen-checkpoint event probe: `scripts/run_cs2_frozen_event_probe.sh`
- Causal event-probe training-seed aggregate: `scripts/summarize_cs2_event_probe_sweep.py`
- GH200 completed-run audit: `scripts/audit_cs2_gh200_sync_control.py`
- GH200 training-seed aggregate: `scripts/summarize_cs2_gh200_sweep.py`
- Completed pilot audit: `scripts/audit_cs2_rebuttal_run.py`
- Fresh single-MIRA endpoint audit: `scripts/audit_cs2_single_confirmatory_endpoint.py`

Every launcher records the code commit/status/patch, resolved Hydra configuration, dataset
selection, checkpoint hashes, environment lock hashes, installed packages, GPU details, and local
JSONL metrics needed to audit the result without W&B.

The completed-pilot auditor fails closed unless the pinned Dust2 selection and split counts,
match-disjointness, clean source provenance, exact timed stage sequence, identical loader settings,
equal 160-frame and ten-action-stream optimizer steps, final time-limit checkpoints, full 520-POV
test cardinality, fixed seed sets, checkpoint hashes, evaluator provenance, and all automatic
evaluation guards verify. The pilot run additionally records
five-second `nvidia-smi` telemetry for peak-memory
disclosure; because telemetry was enabled after the first single-arm checkpoint, single-arm peak
memory from that trace is explicitly labeled as partial, while the shared arm is fully covered.

The GH200 evaluator ends by running its separate fail-closed auditor. It verifies one Slurm job
with four distinct hostnames and rank identities, exactly one scheduler-visible GH200 per node,
common clean commit, three or more node-local loader repeats bound to the confirmatory manifest,
the common frozen loader selection, per-node telemetry, 640 global model-facing frames per
optimizer step, identical fixed update counts and processed frames, absence of safety-cap
termination, spatial routing, fixed validation-rollout cadence, exact final-step checkpoint hashes,
all 69 untouched test rounds (690 POV rows), five evaluation seeds, and complete midpoint and
first-death four-mode action-intervention grids.
After at least three child audits pass, the sweep summarizer requires counterbalanced arm order and
common code/data provenance, then aggregates each nested evaluation-seed mean across training
seeds. The reported independent unit is therefore the training seed; diffusion evaluation seeds
are not incorrectly promoted to independent replications.
