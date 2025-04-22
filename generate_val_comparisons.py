import cv2
import numpy as np
import os
import torch
from models.advanced_fusion_model import AdvancedFusionModel
import torchvision.transforms as transforms
from PIL import Image

def load_image(image_path):
    """加载并预处理图像"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    new_h = (h // 3) * 3
    new_w = (w // 3) * 3
    image = cv2.resize(image, (new_w, new_h))
    
    image = torch.from_numpy(image).float() / 255.0
    image = image.permute(2, 0, 1).unsqueeze(0)
    
    return image, (new_h, new_w)

def save_image(tensor, path, original_size=None):
    """保存图像"""
    image = tensor.squeeze(0).cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if original_size is not None:
        image = cv2.resize(image, (original_size[1], original_size[0]))
    
    cv2.imwrite(path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def create_comparison_image(ir_path, vis_path, fused_path, output_path):
    """创建对比图像"""
    ir_img = cv2.imread(ir_path)
    vis_img = cv2.imread(vis_path)
    fused_img = cv2.imread(fused_path)
    
    if ir_img is None or vis_img is None or fused_img is None:
        print(f"无法读取图像: {ir_path}, {vis_path}, 或 {fused_path}")
        return
    
    h, w = ir_img.shape[:2]
    vis_img = cv2.resize(vis_img, (w, h))
    fused_img = cv2.resize(fused_img, (w, h))
    
    canvas = np.zeros((h, w*3, 3), dtype=np.uint8)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (255, 255, 255)
    
    cv2.putText(ir_img, 'Infrared', (10, 30), font, font_scale, text_color, font_thickness)
    cv2.putText(vis_img, 'Visible', (10, 30), font, font_scale, text_color, font_thickness)
    cv2.putText(fused_img, 'Fused', (10, 30), font, font_scale, text_color, font_thickness)
    
    canvas[:, :w] = ir_img
    canvas[:, w:w*2] = vis_img
    canvas[:, w*2:] = fused_img
    
    cv2.line(canvas, (w, 0), (w, h), (255, 255, 255), 2)
    cv2.line(canvas, (w*2, 0), (w*2, h), (255, 255, 255), 2)
    
    cv2.imwrite(output_path, canvas)
    print(f"已生成对比图像: {output_path}")

def process_validation_set():
    # 设置路径
    val_ir_dir = 'dataset/val/infrared'
    val_vis_dir = 'dataset/val/visible'
    temp_fusion_dir = 'temp_fusion_results'
    comparison_dir = 'validation_comparisons'
    
    # 创建必要的目录
    os.makedirs(temp_fusion_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化模型
    model = AdvancedFusionModel().to(device)
    
    # 加载预训练权重
    if os.path.exists('best_advanced_fusion_model.pth'):
        print("加载预训练权重...")
        checkpoint = torch.load('best_advanced_fusion_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载第 {checkpoint['epoch']} 轮的模型权重")
    else:
        print("未找到预训练权重，使用随机初始化的模型")
    
    # 设置为评估模式
    model.eval()
    
    # 获取所有图像对
    ir_files = sorted([f for f in os.listdir(val_ir_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    vis_files = sorted([f for f in os.listdir(val_vis_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"找到 {len(ir_files)} 对图像")
    
    # 处理每对图像
    for idx, (ir_file, vis_file) in enumerate(zip(ir_files, vis_files)):
        print(f"\n处理第 {idx+1}/{len(ir_files)} 对图像...")
        
        # 构建完整路径
        ir_path = os.path.join(val_ir_dir, ir_file)
        vis_path = os.path.join(val_vis_dir, vis_file)
        
        # 加载图像
        ir_image, ir_size = load_image(ir_path)
        vis_image, vis_size = load_image(vis_path)
        
        # 将图像移到设备上
        ir_image = ir_image.to(device)
        vis_image = vis_image.to(device)
        
        # 进行融合
        with torch.no_grad():
            fused_image = model(ir_image, vis_image)
        
        # 保存融合结果
        fused_path = os.path.join(temp_fusion_dir, f'fused_{idx:04d}.png')
        save_image(fused_image, fused_path, ir_size)
        
        # 创建对比图像
        comparison_path = os.path.join(comparison_dir, f'comparison_{idx:04d}.png')
        create_comparison_image(ir_path, vis_path, fused_path, comparison_path)
    
    print("\n所有图像处理完成！")
    print(f"对比图像保存在: {comparison_dir}")

if __name__ == '__main__':
    process_validation_set() 