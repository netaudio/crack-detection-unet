import torch
import os

print("="*50)
print("诊断信息")
print("="*50)

# 检查是否存在旧权重文件
output_dir = 'output_results'
if os.path.exists(output_dir):
    files = os.listdir(output_dir)
    pth_files = [f for f in files if f.endswith('.pth')]
    if pth_files:
        print(f"发现以下 .pth 文件:")
        for f in pth_files:
            print(f"  - {f}")
            file_path = os.path.join(output_dir, f)
            file_size = os.path.getsize(file_path)
            print(f"    大小: {file_size} bytes")
    else:
        print("未发现 .pth 文件")
else:
    print(f"目录不存在: {output_dir}")

# 测试模型创建
print("\n测试模型创建...")
from eff_fpn_unet import EffFPNUNet

try:
    model = EffFPNUNet(out_channels=1)
    print("✓ 模型创建成功")
    
    # 测试前向传播
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    x = torch.randn(1, 3, 512, 512).to(device)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"✓ 前向传播成功")
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {output.shape}")
    
except Exception as e:
    print(f"✗ 错误: {e}")

print("="*50)