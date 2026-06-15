import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# =========================
# EfficientNet Encoder
# =========================
class EfficientNetEncoder(nn.Module):
    """
    固定输出5个stage特征
    """
    def __init__(self):
        super().__init__()
        eff = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.features = eff.features

    def forward(self, x):
        feats = []

        # EfficientNet-B0 常见stage索引（稳定写法）
        target_layers = [1, 2, 3, 5, 7]

        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in target_layers:
                feats.append(x)

        return feats  # [f1, f2, f3, f4, f5]


# =========================
# Channel Adapter（稳定版）
# =========================
class ChannelAdapter(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# =========================
# Decoder Block
# =========================
class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# =========================
# EffFPNUNet（稳定版）
# =========================
class EffFPNUNet(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()

        self.encoder = EfficientNetEncoder()

        # EfficientNet-B0 各stage通道（固定！）
        self.align1 = ChannelAdapter(16, 256)
        self.align2 = ChannelAdapter(24, 256)
        self.align3 = ChannelAdapter(40, 256)
        self.align4 = ChannelAdapter(112, 256)
        self.align5 = ChannelAdapter(1280, 256)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # Decoder（通道已经固定匹配）
        self.dec4 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(512, 128)
        self.dec1 = DecoderBlock(384, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[-2:]

        f1, f2, f3, f4, f5 = self.encoder(x)

        p1 = self.align1(f1)
        p2 = self.align2(f2)
        p3 = self.align3(f3)
        p4 = self.align4(f4)
        p5 = self.align5(f5)

        d4 = self.dec4(torch.cat([self.up(p5), p4], dim=1))
        d3 = self.dec3(torch.cat([self.up(d4), p3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), p2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), p1], dim=1))

        out = self.final(d1)

        out = F.interpolate(
            out,
            size=input_size,
            mode="bilinear",
            align_corners=False
        )

        return out


# =========================
# 测试
# =========================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EffFPNUNet(out_channels=1).to(device)

    print("=" * 50)
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    x = torch.randn(2, 3, 512, 512).to(device)

    with torch.no_grad():
        y = model(x)

    print(f"\n输入:  {x.shape}")
    print(f"输出:  {y.shape}")
    print("=" * 50)
    print("OK")