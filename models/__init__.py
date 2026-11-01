from .wavelet import dwt_haar, iwt_haar
from .dysample import DySample
from .fdm import FrequencyDecompositionModule, ChannelAttention
from .boundary_modules import BoundaryAttentionGate, BoundaryRefinement

# FDSUNeXT 依赖 timm/sam2，懒加载避免纯模块测试强依赖
__all__ = ['FDSUNeXT', 'Adapter', 'DoubleConv', 'Up',
           'FrequencyDecompositionModule', 'ChannelAttention',
           'BoundaryAttentionGate', 'BoundaryRefinement',
           'DySample', 'dwt_haar', 'iwt_haar', 'MLPFusionHead']


def __getattr__(name):
    if name in ('FDSUNeXT', 'Adapter', 'DoubleConv', 'Up'):
        from . import fds_unext
        return getattr(fds_unext, name)
    if name == 'MLPFusionHead':
        from .mlp_head import MLPFusionHead
        return MLPFusionHead
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
