import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def calculate_metrics(img1, img2):
    """计算图像质量指标"""
    # 确保图像是float32类型
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    # MSE (均方误差)
    mse = np.mean((img1 - img2) ** 2)
    
    # PSNR (峰值信噪比)
    if mse == 0:
        psnr = 100
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # SSIM (结构相似性)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim = np.mean(ssim_map)
    
    return mse, psnr, ssim

def analyze_fusion_results():
    # 设置路径
    fusion_dir = "fusion_results/fusion_results"
    infrared_dir = "dataset/val/infrared"
    visible_dir = "dataset/val/visible"
    
    # 获取所有融合结果文件
    fusion_files = sorted([f for f in os.listdir(fusion_dir) if f.startswith('fused_')])
    
    # 获取所有原始图像文件
    ir_files = sorted([f for f in os.listdir(infrared_dir) if f.endswith('.png')])
    
    # 初始化指标存储
    # 融合结果与红外图像的指标
    mse_values_fused_ir = []
    psnr_values_fused_ir = []
    ssim_values_fused_ir = []
    
    # 融合结果与可见光图像的指标
    mse_values_fused_vis = []
    psnr_values_fused_vis = []
    ssim_values_fused_vis = []
    
    # 红外与可见光图像之间的指标
    mse_values_ir_vis = []
    psnr_values_ir_vis = []
    ssim_values_ir_vis = []
    
    # 遍历所有图像
    for fusion_file, ir_file in zip(fusion_files, ir_files):
        # 构建文件路径
        ir_path = os.path.join(infrared_dir, ir_file)
        vis_path = os.path.join(visible_dir, ir_file)
        fusion_path = os.path.join(fusion_dir, fusion_file)
        
        try:
            # 读取图像
            ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
            vis_img = cv2.imread(vis_path, cv2.IMREAD_GRAYSCALE)
            fusion_img = cv2.imread(fusion_path, cv2.IMREAD_GRAYSCALE)
            
            if ir_img is None or vis_img is None or fusion_img is None:
                print(f"警告：无法读取图像组 {ir_file}")
                continue
            
            # 调整图像大小以匹配
            ir_img = cv2.resize(ir_img, (fusion_img.shape[1], fusion_img.shape[0]))
            vis_img = cv2.resize(vis_img, (fusion_img.shape[1], fusion_img.shape[0]))
            
            # 计算融合结果与红外图像的指标
            mse_fused_ir, psnr_fused_ir, ssim_fused_ir = calculate_metrics(fusion_img, ir_img)
            mse_values_fused_ir.append(mse_fused_ir)
            psnr_values_fused_ir.append(psnr_fused_ir)
            ssim_values_fused_ir.append(ssim_fused_ir)
            
            # 计算融合结果与可见光图像的指标
            mse_fused_vis, psnr_fused_vis, ssim_fused_vis = calculate_metrics(fusion_img, vis_img)
            mse_values_fused_vis.append(mse_fused_vis)
            psnr_values_fused_vis.append(psnr_fused_vis)
            ssim_values_fused_vis.append(ssim_fused_vis)
            
            # 计算红外与可见光图像之间的指标
            mse_ir_vis, psnr_ir_vis, ssim_ir_vis = calculate_metrics(ir_img, vis_img)
            mse_values_ir_vis.append(mse_ir_vis)
            psnr_values_ir_vis.append(psnr_ir_vis)
            ssim_values_ir_vis.append(ssim_ir_vis)
            
        except Exception as e:
            print(f"处理图像组 {ir_file} 时出错: {str(e)}")
            continue
    
    # 计算平均指标
    print("\n融合结果分析:")
    print("\n1. 融合图像与红外图像的对比:")
    print(f"平均 MSE: {np.mean(mse_values_fused_ir):.4f}")
    print(f"平均 PSNR: {np.mean(psnr_values_fused_ir):.4f} dB")
    print(f"平均 SSIM: {np.mean(ssim_values_fused_ir):.4f}")
    
    print("\n2. 融合图像与可见光图像的对比:")
    print(f"平均 MSE: {np.mean(mse_values_fused_vis):.4f}")
    print(f"平均 PSNR: {np.mean(psnr_values_fused_vis):.4f} dB")
    print(f"平均 SSIM: {np.mean(ssim_values_fused_vis):.4f}")
    
    print("\n3. 红外图像与可见光图像的对比（原始差异）:")
    print(f"平均 MSE: {np.mean(mse_values_ir_vis):.4f}")
    print(f"平均 PSNR: {np.mean(psnr_values_ir_vis):.4f} dB")
    print(f"平均 SSIM: {np.mean(ssim_values_ir_vis):.4f}")
    
    # 保存详细结果到文本文件
    with open('fusion_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write("融合结果分析报告\n")
        f.write("="*50 + "\n\n")
        
        f.write("1. 融合图像与红外图像的对比\n")
        f.write(f"平均 MSE: {np.mean(mse_values_fused_ir):.4f}\n")
        f.write(f"平均 PSNR: {np.mean(psnr_values_fused_ir):.4f} dB\n")
        f.write(f"平均 SSIM: {np.mean(ssim_values_fused_ir):.4f}\n")
        f.write(f"MSE范围: [{min(mse_values_fused_ir):.4f}, {max(mse_values_fused_ir):.4f}]\n")
        f.write(f"PSNR范围: [{min(psnr_values_fused_ir):.4f}, {max(psnr_values_fused_ir):.4f}] dB\n")
        f.write(f"SSIM范围: [{min(ssim_values_fused_ir):.4f}, {max(ssim_values_fused_ir):.4f}]\n\n")
        
        f.write("2. 融合图像与可见光图像的对比\n")
        f.write(f"平均 MSE: {np.mean(mse_values_fused_vis):.4f}\n")
        f.write(f"平均 PSNR: {np.mean(psnr_values_fused_vis):.4f} dB\n")
        f.write(f"平均 SSIM: {np.mean(ssim_values_fused_vis):.4f}\n")
        f.write(f"MSE范围: [{min(mse_values_fused_vis):.4f}, {max(mse_values_fused_vis):.4f}]\n")
        f.write(f"PSNR范围: [{min(psnr_values_fused_vis):.4f}, {max(psnr_values_fused_vis):.4f}] dB\n")
        f.write(f"SSIM范围: [{min(ssim_values_fused_vis):.4f}, {max(ssim_values_fused_vis):.4f}]\n\n")
        
        f.write("3. 红外图像与可见光图像的对比（原始差异）\n")
        f.write(f"平均 MSE: {np.mean(mse_values_ir_vis):.4f}\n")
        f.write(f"平均 PSNR: {np.mean(psnr_values_ir_vis):.4f} dB\n")
        f.write(f"平均 SSIM: {np.mean(ssim_values_ir_vis):.4f}\n")
        f.write(f"MSE范围: [{min(mse_values_ir_vis):.4f}, {max(mse_values_ir_vis):.4f}]\n")
        f.write(f"PSNR范围: [{min(psnr_values_ir_vis):.4f}, {max(psnr_values_ir_vis):.4f}] dB\n")
        f.write(f"SSIM范围: [{min(ssim_values_ir_vis):.4f}, {max(ssim_values_ir_vis):.4f}]\n\n")
        
        f.write("4. 分析结论\n")
        # 计算改进比例
        psnr_improvement = (np.mean(psnr_values_fused_ir) - np.mean(psnr_values_ir_vis))
        ssim_improvement = (np.mean(ssim_values_fused_ir) - np.mean(ssim_values_ir_vis))
        
        f.write(f"与原始图像对比的PSNR提升: {psnr_improvement:.4f} dB\n")
        f.write(f"与原始图像对比的SSIM提升: {ssim_improvement:.4f}\n\n")
        
        if psnr_improvement > 0:
            f.write("融合结果成功提升了图像质量，PSNR有明显改善\n")
        else:
            f.write("融合结果的PSNR需要进一步优化\n")
            
        if ssim_improvement > 0:
            f.write("融合结果保持了良好的结构信息，SSIM有所提升\n")
        else:
            f.write("融合结果的结构保持需要加强\n")

if __name__ == "__main__":
    analyze_fusion_results() 