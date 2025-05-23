# 基于Advanced SwinTransformer的红外与可见光图像融合项目

本项目实现了一个基于Advanced SwinTransformer的红外与可见光图像融合系统，使用先进的Transformer架构来生成高质量的融合图像。该模型能够有效地融合红外图像的热目标信息和可见光图像的细节信息，同时利用Transformer的全局建模能力提升融合效果。

## 项目结构

```
.
├── models/
│   └── advanced_fusion_model.py    # Advanced SwinTransformer融合模型定义
├── configs/
│   └── config1.yaml                # 模型配置文件
├── analysis/                       # 评估工具
│   ├── Python/
│   │   ├── metric_calculation/     # 指标计算
│   │   └── metric_analysis/        # 结果分析
├── dataset/                        # 数据集
│   ├── train/                      # 训练集
│   │   ├── infrared/              # 红外图像
│   │   └── visible/               # 可见光图像
│   └── val/                       # 验证集
│       ├── infrared/              # 红外图像
│       └── visible/               # 可见光图像
├── validation_comparisons/         # 验证集对比结果
├── train_advanced_fusion.py        # 训练脚本
├── test_advanced_fusion.py         # 测试脚本
├── generate_val_comparisons.py     # 对比图像生成工具
└── requirements.txt                # 依赖包列表
```

## 环境要求

- Python 3.8+
- PyTorch 1.8+
- OpenCV
- NumPy
- timm
- torchvision

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

```bash
python train_advanced_fusion.py
```

训练过程会：
- 自动保存最佳模型权重到 `best_advanced_swin_fusion_model.pth`
- 每10轮保存一次检查点到 `advanced_swin_fusion_model_epoch_*.pth`

### 2. 测试模型

```bash
python test_advanced_fusion.py
```

### 3. 生成验证集对比图像

```bash
python generate_val_comparisons.py
```

### 4. 评估融合结果

```bash
# 计算单张图像的评估指标
python analysis/Python/metric_calculation/main.py --input path/to/fused/image.png --reference path/to/reference/image.png

# 批量评估
python analysis/Python/metric_calculation/batch_calculate.py --input-dir path/to/fused/images --reference-dir path/to/reference/images

# 生成分析报告
python analysis/Python/metric_analysis/analyze.py --results-dir path/to/results
```

## 模型说明

Advanced SwinTransformer融合模型采用以下技术：
- 基于SwinTransformer的特征提取：利用窗口注意力机制提取多尺度特征
- 层次化特征融合：通过多级Transformer块实现特征交互
- 跨分支交互：促进红外和可见光特征的互补
- 上采样重建：使用转置卷积进行图像重建
- 细节增强：提升融合图像的细节表现

## 评估指标

项目支持以下评估指标：
- 结构相似性（SSIM）
- 峰值信噪比（PSNR）
- 互信息（MI）
- 视觉信息保真度（VIF）
- 边缘保持度（EPI）

## 结果展示

验证集对比结果保存在 `validation_comparisons` 目录下，包含60对图像的融合结果对比。每张对比图包含：
- 左侧：红外图像
- 中间：可见光图像
- 右侧：融合结果

## 注意事项

1. 训练前请确保：
   - 数据集准备完整
   - GPU内存充足（建议至少8GB显存）
   - 配置文件参数正确

2. 训练建议：
   - 使用GPU加速训练
   - 确保输入图像大小一致
   - 建议使用高质量的红外和可见光图像对
   - 可以使用混合精度训练加速

3. 评估建议：
   - 使用多种评估指标综合判断
   - 结合主观视觉评估
   - 注意不同场景下的表现差异

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@article{advanced-swin-fusion,
  title={Advanced Infrared and Visible Image Fusion with Swin Transformer},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```
#   A d v a n c e d _ S w i n _ F u s i o n  
 