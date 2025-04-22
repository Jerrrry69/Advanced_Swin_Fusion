import torch
from .fusion_model import FusionModel

def build_model(config):
    """
    构建融合模型
    Args:
        config: 配置参数
    Returns:
        model: 融合模型
    """
    model = FusionModel(
        in_channels=config.MODEL.FUSION.IN_CHANNELS,
        out_channels=config.MODEL.FUSION.OUT_CHANNELS,
        hidden_channels=config.MODEL.FUSION.HIDDEN_CHANNELS,
        num_blocks=config.MODEL.FUSION.NUM_BLOCKS
    )
    
    return model 