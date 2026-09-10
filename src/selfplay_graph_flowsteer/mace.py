"""Deprecated import facade for historical audit scripts, not a production router.

Canvas, Adaptive and application constructors reject MACE injection. Old files
remain readable; new execution and training use Director SET_MODEL exclusively.
"""

from .legacy.mace import (  # noqa: F401
    FEATURE_DIM,
    MODEL_FEATURE_DIM,
    MACEModelRouter,
    MACEPeerSelector,
    ModelBanditState,
    ModelSelectionRecord,
    PeerBanditState,
    PeerSelectionRecord,
    model_context_features,
    relational_features,
    semantic_agent_identity,
)
