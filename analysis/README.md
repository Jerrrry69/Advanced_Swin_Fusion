# 图像融合分析模块

本模块提供了图像融合结果的评估和分析工具。

## 目录结构

```
analysis/
├── Python/
│   ├── metric calculation/    # 指标计算工具
│   └── metric analysis/       # 指标分析工具
└── results.txt               # 分析结果输出
```

## 功能说明

### 1. 指标计算 (metric calculation)

计算融合图像的各项评估指标：
- 结构相似性指数 (SSIM)
- 峰值信噪比 (PSNR)
- 互信息 (MI)
- 视觉信息保真度 (VIF)

### 2. 指标分析 (metric analysis)

分析不同融合方法的性能对比：
- 指标可视化
- 统计分析
- 性能对比报告

## 使用方法

1. 计算单张图像的指标：
```bash
python metric_calculation/main.py --input path/to/fused/image.png --reference path/to/reference/image.png
```

2. 批量计算指标：
```bash
python metric_calculation/batch_calculate.py --input-dir path/to/fused/images --reference-dir path/to/reference/images
```

3. 生成分析报告：
```bash
python metric_analysis/analyze.py --results-dir path/to/results
```

## 输出说明

- 指标计算结果保存在 `results.txt` 文件中
- 分析报告和图表保存在 `analysis/output` 目录下 