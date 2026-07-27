# Entry-point scripts

Hydra applications for training, evaluation, and serving. Each reads its config from `configs/`
(override any key as `key=value`) and runs single-GPU, or multi-GPU under `torchrun`.

- `train_codec.py` — train the video codec.
- `train_world_model.py` — train the world model (single-player, or add
  `model=multi_wrapper_world_model dataset.n_players=4` for 4-player).
- `eval_world_model_offline.py` — offline evaluation of a trained world model (validation loss +
  rollout metrics) from a checkpoint.
- `bench_wm_speed.py` — micro-benchmark world-model rollout speed.
- `bench_cs2_dataloader.py` — benchmark the exact Dust2 MIRA input contract across worker settings
  and fail closed unless the fixed first-round single/synchronized video and action tensors match.
- `prepare_cs2_confirmatory_split.py` — freeze or verify the post-pilot, match-atomic Dust2
  confirmatory split by salted hash only, quarantining the already consumed release test.
- `diagnose_cs2_multi_action_routing.py` — quantify how much global or spatial multiplayer routing
  preserves a deterministic cross-POV action intervention before the diffusion transformer.
- `run_cs2_spatial_routing_preflight.sh` — run the frozen one-hour G7e validation-only engineering
  gate for spatial player-aligned action routing.
- `assess_cs2_spatial_routing_preflight.py` — enforce the preregistered routing and action-loss
  thresholds without consulting the new confirmatory test.
- `extract_cs2_future_event_features.py` — read only pre-horizon pixels/actions and export the
  frozen checkpoint's last causal context representation without editing the MIRA architecture.
- `run_cs2_frozen_event_probe.sh` — compare identical synchronized contexts represented by the
  single, synchronized-trained, and matched-information cross-round-trained checkpoints using the
  public CounterStrike-1K future-event probe. Set both `CS1K_SYNCHRONIZED_CHECKPOINT` and
  `CS1K_SHUFFLED_CHECKPOINT` for an exact common-step comparison; otherwise the launcher uses each
  arm's latest checkpoint. `CS1K_CODEC_CHECKPOINT` may relocate the immutable codec when a frozen
  world-model config contains an absolute path from its original training host.
- `summarize_cs2_event_probe_sweep.py` — verify every event checkpoint against its passing GH200
  audit, average repeated probe-head fits within each frozen model pair, and aggregate only across
  independent world-model training seeds.
- `submit_cs2_gh200_sweep.sh` — from a Slurm login node, submit the complete three-seed
  full-node training, causal-event, and final aggregation dependency graph with one command. Each
  training seed uses all four GH200s on one Clariden node with four local DDP processes. This
  lower-level submitter assumes its runtime has already been prepared.
- `prepare_and_submit_cs2_clariden_sweep.sh` — preferred Clariden entry point. It layers the
  required packages over CSCS's pinned ARM64 `pytorch/v2.8.0:v1` uenv and then invokes the complete
  dependency submitter. Copy `clariden_sync_sweep.env.example` outside the checkout, fill the
  shared paths and Slurm account, then run
  `bash scripts/prepare_and_submit_cs2_clariden_sweep.sh /path/to/clariden_sync_sweep.env`.
- `setup_cs2_clariden_uenv.sh` — idempotent environment preparation invoked by the preferred
  wrapper. It verifies ARM64 and PyTorch 2.8, pins TorchCodec 0.7, installs both clean public
  checkouts into one uenv-layered venv, and records the exact package/runtime provenance.
- `submit_cs2_clariden_dataset_stage.sh` — submit one resumable Clariden batch job that downloads
  only the 116 pinned 360p Dust2 WebDataset shards, regenerates the frozen confirmatory split,
  materializes all five payloads for 9,410 samples, and verifies the publication hashes. Its worker
  is `run_cs2_clariden_dataset_stage.sh`; the pinned Hugging Face selection is implemented by
  `download_cs2_dust2_subset.py`.
- `stage_cs2_frozen_endpoints.sh` — atomically install the immutable codec and single-MIRA
  comparison endpoint from a short-lived download URL. It verifies the bundle member list, bundle
  SHA-256, and all four extracted file hashes against
  `configs/frozen_dust2_endpoints_v1.json`, and refuses conflicting destinations.
- `run_cs2_frozen_event_probe_slurm_seed.sh` — dependent one-GPU event job for one passing
  fixed-update child audit, with explicit common-step checkpoints, frozen single-MIRA hash, and a
  job-private prepared Torch Hub cache.
- `prepare_and_submit_cs2_clariden_event_recovery.sh` — event-only Clariden recovery after
  successful checkpoint-only evaluation. It archives failed partial event roots, reuses all
  training/evaluation and completed event artifacts, optionally retries only
  `CS1K_EVENT_RECOVERY_SEEDS`, and runs final aggregation after the selected retries pass. Its
  lower-level submitter is `submit_cs2_gh200_event_recovery.sh`.
- `run_cs2_gh200_sweep_finalize.sh` — dependent fail-closed aggregation job; writes
  `sweep_summary.json` and `event_probe_sweep_summary.json` only after all child jobs pass.
- `render_cs2_rebuttal_report.py` — render deterministic Markdown tables only after the complete
  Dust2 run audit passes and all summaries match the audited checkpoint and seed contracts.
- `validate_cs2_loader_selection.py` — freeze a GH200 loader configuration only from at least three
  clean-source, transfer-inclusive, exact-contract synchronized/shuffled benchmark repeats.
- `watch_cs2_strict_audit_report.sh` — after the pinned pilot audit passes, run the stronger
  versioned identity audit and render a hash-bound Markdown report without manual intervention.

See [`configs/README.md`](../configs/README.md) for the config layout and the top-level
[`README.md`](../README.md) for example commands.
