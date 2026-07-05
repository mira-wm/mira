# MIRA

MIRA (Multiplayer Interactive World Models with Representation Autoencoders) is a real-time neural
world model of Rocket League: instead of a game engine, a network generates the video frame by frame
from the players' inputs, so four people can play a full 2v2 match inside the model at once. Play it
live at https://mira-wm.com.

A collaboration between General Intuition (https://www.generalintuition.com) and Kyutai
(https://kyutai.org), with the dataset, models, and training and inference code all open-sourced.

This repository contains the code accompanying our technical report,
*MIRA: Multiplayer Interactive World Models with Representation Autoencoders*. The report introduces
the first multiplayer world model for a highly dynamic environment governed by complex physical
interactions: a 5-billion-parameter latent diffusion model, trained on 10,000 hours of gameplay, that
conditions on all four players' action streams to faithfully simulate 2v2 Rocket League matches in
real time. Everything needed to reproduce it lives here — the dataset loader, the video codec, the
action-conditioned world model, and the training and inference pipelines.

Code: https://github.com/mira-wm/mira
Dataset: https://huggingface.co/datasets/kyutai/rocket-science

![Four synchronised player views with the top-down arena radar](docs/hero.png)

Code structure:

```
src/mira/
  data/         # dataset loader: 4-player clips, actions, events, game state (+ viz)
  ml/           # shared model building blocks
  codec/        # video codec (frames <-> latents)
  world_model/  # the action-conditioned world model (+ 4-player wrapper)
  training/     # training infrastructure: EMA, schedules, checkpoints, metrics, logging
  inference/    # autoregressive rollout of the world model
configs/        # Hydra configs for the training/eval entry points
scripts/        # train_codec, train_world_model, eval_world_model_offline, bench
```

## The data

One sample is a ~4 s window of a 2v2 match with all four players' synchronised views. Each frame
carries, per perspective: the **video frame**, the **keyboard action** (multi-hot over a fixed key
set), the discrete **game events** on a shared master clock, and the per-frame **game state** (ball,
cars, score). The full dataset is on the
[HuggingFace Hub](https://huggingface.co/datasets/kyutai/rocket-science).

```python
from mira.data import RocketLeagueDataset

ds = RocketLeagueDataset.from_hub("kyutai/rocket-science", split="test")  # or .from_local("/path/to/data")
clip = ds.load_match(ds.match_ids()[0], clip_len=16, target_fps=10)[0]

clip.frames      # (P, T, C, H, W) uint8, one decoded video per perspective
clip.actions     # (P, T, 9) int32, multi-hot keyboard, frame-aligned
clip.events      # discrete game events overlapping this window
clip.physics     # per perspective, per-frame game state (ball / cars / score)

for clip in ds.iter_clips(clip_len=16, target_fps=10, exclude_replays=True):  # streaming
    ...

ds = RocketLeagueDataset.from_hub("kyutai/rocket-science", split="test", shards=1)  # one shard only
```

The full `test` split is ~74 GB across 31 shards; pass `shards=N` to `from_hub` to pull only the
first `N` shards (and the matches they contain) for a quick smoke test.

To explore it interactively, launch the guided tour and pick a match and clip; every view then
re-renders (a synchronised 4-player grid with a keyboard overlay, plus a top-down arena radar and
HUD):

```bash
pixi run explore     # read-only app view
pixi run edit        # editable (headless on :2718; SSH-tunnel to view)
```

See the [dataset card](docs/dataset_card.md) for the full data model.

## Training & inference

The model code, the Hydra entry points under `scripts/`, and the configs under `configs/` install
with the `train` extra (`pip install mira[train]`, included by `pixi run setup`). The code
targets **torch >= 2.8**; `pixi run setup` installs a matching build for you.

### DINOv3 backbone weights (codec training only)

The codec's frozen encoder is a **DINOv3-L/16** backbone. Its *code* is pulled from `torch.hub`
(`facebookresearch/dinov3`, needs network on the first build), but the *pretrained weights are gated
by Meta* and are **not** downloaded automatically. To train the codec, fetch them yourself:

1. Accept the license and download `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` from the
   [DINOv3 downloads page](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/).
2. Put it in a directory and point the codec at it:
   ```bash
   export RS_DINO_WEIGHTS_DIR=/path/to/dinov3/weights   # dir holding the .pth above
   ```

Without `RS_DINO_WEIGHTS_DIR`, `train_codec.py` raises a `FileNotFoundError` saying the same.
World-model training and inference don't need it; a codec checkpoint already carries the backbone.

Entry points are Hydra apps: override any config key as `key=value`. Multi-GPU runs go through
`torchrun`, and the distributed setup no-ops for a single process, so the commands below also
run single-GPU.

```bash
# Train the codec
python scripts/train_codec.py dataset.train_index=/path/to/train dataset.test_index=/path/to/test

# Train the single-player world model
python scripts/train_world_model.py \
    model.architecture.config.codec_checkpoint=/path/to/codec_ckpt \
    dataset.train_index=/path/to/train dataset.test_index=/path/to/test

# Train the 4-player world model, warm-started from a single-player checkpoint
python scripts/train_world_model.py model=multi_wrapper_world_model dataset.n_players=4 \
    model.architecture.config.wm_config.codec_checkpoint=/path/to/codec_ckpt \
    run.finetune_from=/path/to/single_player_ckpt \
    dataset.train_index=/path/to/train dataset.test_index=/path/to/test

# Offline evaluation (validation loss + rollout metrics); the model class is read from the saved
# config. The checkpoint may be a local path or a W&B run URL. The Frechet metrics need the `eval`
# extra (pip install mira[eval]); without it the eval logs and skips them.
python scripts/eval_world_model_offline.py /path/to/checkpoint-1000/checkpoint.pth
```

## Development

The project uses [pixi](https://pixi.sh) for its environment: it provides the Python interpreter and
the FFmpeg build that torchcodec needs to decode video, with the Python packages installed editable
on top via uv.

```bash
pixi run setup      # create the environment and install mira + extras (GPU/CUDA torch)
pixi run setup-cpu  # same, but CPU-only torch (machines without an NVIDIA GPU)
pixi run test       # run the test suite
pixi run verify     # format, lint, typecheck, and test
```

`setup` installs a CUDA (cu128) build of torch so training runs on the GPU out of the box; use
`setup-cpu` when there's no GPU. Video is always decoded on CPU via torchcodec's `+cpu` build (the
CUDA-decode wheel needs CUDA NPP runtime libraries absent off-GPU, so it fails to load there), which
the setup tasks pin for you.

Pip-only users who already have a compatible FFmpeg can skip pixi and
`pip install mira[decode,viz]` into their own environment.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
