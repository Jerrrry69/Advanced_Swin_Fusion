import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import SwinTransformerBlock
from timm.models.layers import PatchEmbed, Mlp

class AdvancedSwinFusionModel(nn.Module):
    def __init__(self, img_size=256, patch_size=4, in_chans=3, embed_dim=96,
                 depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True):
        super(AdvancedSwinFusionModel, self).__init__()
        
        # 红外图像特征提取
        self.ir_patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)
        
        # 可见光图像特征提取
        self.vis_patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)
        
        # 特征融合层
        self.fusion_blocks = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = nn.Sequential(
                *[SwinTransformerBlock(
                    dim=embed_dim * (2 ** i_layer),
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_rate,
                    norm_layer=norm_layer)
                  for i in range(depths[i_layer])])
            self.fusion_blocks.append(layer)
        
        # 上采样和重建
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(embed_dim * 8, embed_dim * 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(embed_dim * 4, embed_dim * 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(embed_dim * 2, embed_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, 3, kernel_size=1)
        )
        
        # 细节增强
        self.detail_enhance = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1)
        )

    def forward(self, x_ir, x_vis):
        # 提取特征
        ir_feat = self.ir_patch_embed(x_ir)
        vis_feat = self.vis_patch_embed(x_vis)
        
        # 特征融合
        for block in self.fusion_blocks:
            ir_feat = block(ir_feat)
            vis_feat = block(vis_feat)
            # 特征交互
            ir_feat = ir_feat + vis_feat
            vis_feat = vis_feat + ir_feat
        
        # 上采样重建
        fused = self.upsample(ir_feat)
        
        # 细节增强
        detail = self.detail_enhance(fused)
        output = fused + 0.1 * detail
        
        return torch.sigmoid(output) 