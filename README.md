## MIRA: Multiplayer Interactive World Models with Representation Autoencoders

[![Live demo](https://img.shields.io/badge/demo-mira--wm.com-0091FF.svg)](https://mira-wm.com)&nbsp;
[![Paper](https://img.shields.io/badge/paper-technical%20report-b31b1b.svg)](https://arxiv.org/abs/2607.05352)&nbsp;
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20dataset-rocket--science-yellow.svg)](https://huggingface.co/datasets/kyutai/rocket-science)&nbsp;
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

https://github.com/user-attachments/assets/ca82b521-3e53-4f1b-8b27-9126a49aa5ea

MIRA is a real-time world model of Rocket League: a 5B parameters
latent diffusion model generates the video frame by frame from all four players' actions, so a full 2v2
match can be played inside the model at 20 FPS on a single GPU. Play it live at
[mira-wm.com](https://mira-wm.com).

This is the official code release for the technical report:<br>
[MIRA: Multiplayer Interactive World Models with Representation Autoencoders](https://github.com/mira-wm/mira), a collaboration between<br>
[General Intuition](https://www.generalintuition.com), [Kyutai](https://kyutai.org), and [Epic Games](https://www.epicgames.com/site/home).

<img width="1491" height="533" alt="architecture" src="https://github.com/user-attachments/assets/4f0fa7c7-aa27-48ed-9b08-4ebaade65f4d" />

```
@article{mira2026,
  title={{MIRA}: Multiplayer Interactive World Models with Representation Autoencoders},
  author={Hu, Anthony and Volhejn, V{\'a}clav and Ramanana Rahary, Adrien and Mulder, Chris and Makkar, Aditya and Liao, Alyx and Royer, Am{\'e}lie and Orsini, Manu and Jelley, Adam and Alonso, Eloi and Laurent, Florian and Nor{\'e}n, Fredrik and Swingos, James and H{\"u}nermann, Jan and Rollins, Kent and Hosseini, Lucas and Le Cauchois, Matthieu and Peter, Maxim and de Witte, Pim and Brown, Tim and Micheli, Vincent and B{\"o}hle, Moritz and de Marmiesse, Gabriel and Sharmanska, Viktoriia and Specia, Lucia and Black, Michael and P{\'e}rez, Patrick},
  year={2026},
  note={Technical Report},
}
```

### Installation

```bash
pixi run setup   # one-time: creates the environment and installs MIRA (requires an NVIDIA GPU)
pixi run test    # run the test suite
```

Requires [pixi](https://pixi.sh) and torch >= 2.8 (installed for you).

### Dataset

Each sample is a 4-second window of a 2v2 match, captured from all four players' synchronised views.
For every frame of every view, the dataset provides the video frame, the player's keyboard action,
and the game state (ball, cars, and score). The full dataset is on
the [HuggingFace Hub](https://huggingface.co/datasets/kyutai/rocket-science) (see the
[dataset card](docs/dataset_card.md)).

```python
from mira.data import RocketScienceDataset

# Download the test split. shards=1 pulls just one shard for a quick look; omit it for the full split.
ds = RocketScienceDataset.from_hub("kyutai/rocket-science", split="test", shards=1)

# Load the clips of the first match (16 frames each, sampled at 20 FPS) and take the first one.
clips = ds.load_match(ds.match_ids()[0], clip_len=16, target_fps=20)
clip = clips[0]

# A clip holds P=4 synchronised player views over T=16 frames:
clip.frames    # tensor (P, T, C, H, W) uint8 — the rendered video, one per player
clip.actions   # tensor (P, T, 9)       int32 — multi-hot keyboard state, aligned to each frame
clip.events    # list of game events overlapping this window (goals, boost pickups, ...)
clip.physics   # per player, per frame: game state (ball, cars, score)
```

Explore it interactively (4-player grid, keyboard overlay, top-down overview of the game using the game state):

```bash
pixi run explore
```

### Training

Entry points under `scripts/` are Hydra apps — override any config key as `key=value`. Multi-GPU runs
go through `torchrun`; single-GPU works too.

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
```

Codec training uses a frozen **DINOv3-L/16** encoder whose weights are gated by Meta. Download
`dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` from the
[DINOv3 page](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) and set
`RS_DINO_WEIGHTS_DIR=/path/to/weights`. World-model training and inference don't need it.

### Evaluation

```bash
python scripts/eval_world_model_offline.py /path/to/checkpoint-1000/checkpoint.pth
```

### License

Apache License 2.0 — see [LICENSE](LICENSE).
