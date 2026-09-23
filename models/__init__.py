# -*- coding: utf-8 -*-
"""
Package models pour l'architecture Topo-Multimodal.
"""

from models.encoders import (
    ImageEncoder,
    TextEncoder,
    AudioEncoder,
    VideoEncoder,
    ModalityProjector
)
from models.latent_bank import (
    SemanticLatentBank,
    SharedPrivateFactorization,
    AdaptiveFusion
)
from models.topology import (
    AdaptiveLatentGraph,
    LearnedFiltration,
    PersistentHomology,
    TopologicalEncoder
)
from models.attention import (
    TopologicalAttention,
    TopologicalTransformerBlock
)
from models.robustness import (
    RobustnessPerturbationModule
)
from models.multimodal_framework import (
    TopoMultimodalFramework
)

__all__ = [
    "ImageEncoder",
    "TextEncoder",
    "AudioEncoder",
    "VideoEncoder",
    "ModalityProjector",
    "SemanticLatentBank",
    "SharedPrivateFactorization",
    "AdaptiveFusion",
    "AdaptiveLatentGraph",
    "LearnedFiltration",
    "PersistentHomology",
    "TopologicalEncoder",
    "TopologicalAttention",
    "TopologicalTransformerBlock",
    "RobustnessPerturbationModule",
    "TopoMultimodalFramework"
]
