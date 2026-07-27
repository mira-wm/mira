"""mira.codec: the RAEv2 temporal-downsampling video codec.

Public API:
    VideoCodecConfig, RAEEncoderConfig, ViTDecoderConfig, StridedConvBottleneckConfig — configs
    VideoCodec, VideoCodecOutputs — the codec module and its forward outputs
    RAEEncoder, RAEEncoderOutputs — frozen-DINOv3 + strided-conv bottleneck encoder
    ViTVideoDecoder — ViT video decoder
    DinoModel, DinoPerceptualLoss, DINO_DIM — DINO backbone loading + perceptual loss
    CodecLoss, CodecLossWeights — the codec training loss and its configuration
"""

from .codec_model import VideoCodec, VideoCodecOutputs
from .config import (
    RAEEncoderConfig,
    StridedConvBottleneckConfig,
    VideoCodecConfig,
    ViTDecoderConfig,
)
from .dino import DINO_DIM, DinoModel, DinoPerceptualLoss
from .loss import CodecLoss, CodecLossWeights
from .rae_encoder import RAEEncoder, RAEEncoderOutputs
from .vit_decoder import ViTVideoDecoder

__all__ = [
    "VideoCodecConfig",
    "RAEEncoderConfig",
    "ViTDecoderConfig",
    "StridedConvBottleneckConfig",
    "VideoCodec",
    "VideoCodecOutputs",
    "RAEEncoder",
    "RAEEncoderOutputs",
    "ViTVideoDecoder",
    "DinoModel",
    "DinoPerceptualLoss",
    "DINO_DIM",
    "CodecLoss",
    "CodecLossWeights",
]
