"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""


from .deim import DEIM

from .matcher import HungarianMatcher
from .hybrid_encoder import HybridEncoder
from .dfine_decoder import DFINETransformer
from .rtdetrv2_decoder import RTDETRTransformerv2

from .postprocessor import PostProcessor
from .deim_criterion import DEIMCriterion
from .sar_boundary import BoundaryConsistencyMap, ShapeAdaptiveScoring
from .scatter_clutter_suppression import ScatterAwareClutterSuppression
from .detail_preservation import DetailPreservationModule
from .adaptive_feature_selection import AdaptiveFeatureSelection, MultiScaleEnhancement
from .scattering_pattern_isolation import ScatteringPatternIsolation, SPILoss
from .scattering_physics import SPIModule
from .background_rejection import SimpleBackgroundRejection, SBRLoss
from .boundary_refinement import BoundaryRefinementLoss, BRLCriterion
from .scale_aware_refinement import ScaleAwareRefinement, SARLoss
