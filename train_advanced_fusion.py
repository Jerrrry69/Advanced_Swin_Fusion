import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
from models.advanced_fusion_model import AdvancedSwinFusionModel

class ImagePairDataset(Dataset):
    def __init__(self, ir_dir, vis_dir, transform=None):
        self.ir_dir = ir_dir
        self.vis_dir = vis_dir
        self.transform = transform
        self.image_pairs = []
        
        # 获取所有图像对
        ir_files = sorted([f for f in os.listdir(ir_dir) if f.endswith('.png')])
        vis_files = sorted([f for f in os.listdir(vis_dir) if f.endswith('.png')])
        
        for ir_file, vis_file in zip(ir_files, vis_files):
            self.image_pairs.append((ir_file, vis_file))
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        ir_file, vis_file = self.image_pairs[idx]
        
        # 加载图像
        ir_path = os.path.join(self.ir_dir, ir_file)
        vis_path = os.path.join(self.vis_dir, vis_file)
        
        ir_image = Image.open(ir_path).convert('RGB')
        vis_image = Image.open(vis_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            ir_image = self.transform(ir_image)
            vis_image = self.transform(vis_image)
        
        return ir_image, vis_image

class FusionLoss(nn.Module):
    def __init__(self):
        super(FusionLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, fused, ir, vis):
        # 结构相似性损失
        ssim_ir = 1 - self.ssim(fused, ir)
        ssim_vis = 1 - self.ssim(fused, vis)
        
        # 像素级损失
        l1_ir = self.l1_loss(fused, ir)
        l1_vis = self.l1_loss(fused, vis)
        
        # 梯度损失
        grad_ir = self.gradient_loss(fused, ir)
        grad_vis = self.gradient_loss(fused, vis)
        
        # 总损失
        loss = (ssim_ir + ssim_vis) + 0.5 * (l1_ir + l1_vis) + 0.5 * (grad_ir + grad_vis)
        return loss
    
    def gradient_loss(self, x, y):
        """计算梯度损失"""
        x_grad = self._gradient(x)
        y_grad = self._gradient(y)
        return self.mse_loss(x_grad, y_grad)
    
    def _gradient(self, x):
        """计算图像梯度"""
        # 使用Sobel算子计算梯度
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
        
        # 对每个通道分别计算梯度
        grad_x = F.conv2d(x, sobel_x.repeat(x.size(1), 1, 1, 1), padding=1, groups=x.size(1))
        grad_y = F.conv2d(x, sobel_y.repeat(x.size(1), 1, 1, 1), padding=1, groups=x.size(1))
        
        # 合并梯度
        return torch.cat([grad_x, grad_y], dim=1)
    
    def ssim(self, x, y, window_size=11):
        """计算SSIM"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_x = F.avg_pool2d(x, window_size, stride=1, padding=window_size//2)
        mu_y = F.avg_pool2d(y, window_size, stride=1, padding=window_size//2)
        
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y
        
        sigma_x = F.avg_pool2d(x * x, window_size, stride=1, padding=window_size//2) - mu_x_sq
        sigma_y = F.avg_pool2d(y * y, window_size, stride=1, padding=window_size//2) - mu_y_sq
        sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=window_size//2) - mu_xy
        
        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2))
        return ssim_map.mean()

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=50):
    model.train()
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        total_loss = 0
        batch_count = 0
        
        print(f"开始第 {epoch+1} 轮训练...")
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (ir_images, vis_images) in enumerate(progress_bar):
            # 将数据移到GPU
            ir_images = ir_images.to(device)
            vis_images = vis_images.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            fused_images = model(ir_images, vis_images)
            
            # 计算损失
            loss = criterion(fused_images, ir_images, vis_images)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            # 更新进度条
            avg_loss = total_loss / batch_count
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # 计算平均损失
        epoch_loss = total_loss / batch_count
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {epoch_loss:.4f}')
        
        # 保存最佳模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'best_advanced_swin_fusion_model.pth')
            print(f'保存最佳模型，损失值: {best_loss:.4f}')
        
        # 每10个epoch保存一次检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, f'advanced_swin_fusion_model_epoch_{epoch+1}.pth')
            print(f'保存检查点: advanced_swin_fusion_model_epoch_{epoch+1}.pth')

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # 创建数据集和数据加载器
    train_dataset = ImagePairDataset(
        ir_dir='dataset/train/infrared',
        vis_dir='dataset/train/visible',
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4
    )
    
    # 初始化模型
    model = AdvancedSwinFusionModel().to(device)
    
    # 初始化损失函数和优化器
    criterion = FusionLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.05)
    
    # 训练模型
    train_model(model, train_loader, criterion, optimizer, device)

if __name__ == '__main__':
    main() 