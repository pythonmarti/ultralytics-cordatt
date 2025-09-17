# ultralitics/nn/modules/__init__.py

# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# """
# Ultralytics neural network modules.
# """

from .block import (
    C1, C2, C2PSA, C3, C3TR, CIB, DFL, ELAN1, PSA, SPP, SPPELAN, SPPF, A2C2f, AConv, ADown,
    Attention, BNContrastiveHead, Bottleneck, BottleneckCSP, C2f, C2fAttn, C2fCIB, C2fPSA,
    C3Ghost, C3k2, C3x, CBFuse, CBLinear, ContrastiveHead, GhostBottleneck, HGBlock, HGStem,
    ImagePoolingAttn, MaxSigmoidAttnBlock, Proto, RepC3, RepNCSPELAN4, RepVGGDW, ResNetLayer,
    SCDown, TorchVision, C3CA
)
from .conv import (
    CoordAtt, # <-- AÑADIDO
    Concat, Conv, Conv2, ConvTranspose, DWConv, DWConvTranspose2d, Focus, GhostConv, Index,
    LightConv, RepConv
)
from .head import (
    OBB, Classify, Detect, LRPCHead, Pose, RTDETRDecoder, Segment, WorldDetect, YOLOEDetect,
    YOLOESegment, v10Detect
)
from .transformer import (
    AIFI, MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer, LayerNorm2d,
    MLPBlock, MSDeformAttn, TransformerBlock, TransformerEncoderLayer, TransformerLayer
)

# SE ELIMINARON LAS REFERENCIAS A CBAM Y SE AÑADIÓ CoordAtt
__all__ = (
    "Conv", "Conv2", "LightConv", "RepConv", "DWConv", "DWConvTranspose2d", "ConvTranspose", "Focus",
    "GhostConv", "CoordAtt", "Concat", "TransformerLayer", "TransformerBlock", "MLPBlock",
    "LayerNorm2d", "DFL", "HGBlock", "HGStem", "SPP", "SPPF", "C1", "C2", "C3", "C2f", "C3k2",
    "SCDown", "C2fPSA", "C2PSA", "C2fAttn", "C3x", "C3TR", "C3Ghost", "GhostBottleneck",
    "Bottleneck", "BottleneckCSP", "Proto", "Detect", "Segment", "Pose", "Classify",
    "TransformerEncoderLayer", "RepC3", "RTDETRDecoder", "AIFI", "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer", "MSDeformAttn", "MLP", "ResNetLayer", "OBB",
    "WorldDetect", "YOLOEDetect", "YOLOESegment", "v10Detect", "LRPCHead", "ImagePoolingAttn",
    "MaxSigmoidAttnBlock", "ContrastiveHead", "BNContrastiveHead", "RepNCSPELAN4", "ADown",
    "SPPELAN", "CBFuse", "CBLinear", "AConv", "ELAN1", "RepVGGDW", "CIB", "C2fCIB", "Attention",
    "PSA", "TorchVision", "Index", "A2C2f", "C3CA"
)