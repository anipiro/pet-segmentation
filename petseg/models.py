"""Model zoo. Every model maps (N, 3, H, W) -> logits (N, 2, H, W).

  unet_scratch     classic U-Net (Ronneberger et al. 2015), random init, 16 base
                   channels instead of 64 to fit a CPU training budget.
                   Baseline: what do we get without any pretraining?
  unet_resnet18    U-Net decoder on an ImageNet-pretrained ResNet-18 encoder.
  lraspp_mbv3      torchvision LR-ASPP, MobileNetV3-Large backbone (ImageNet).
                   Built for mobile; the smallest/fastest candidate.
  deeplabv3_mbv3   torchvision DeepLabV3, MobileNetV3-Large backbone (ImageNet).
                   Atrous pyramid pooling for multi-scale context.

For the torchvision models only the backbone is pretrained (ImageNet
classification); the segmentation heads start from scratch. We deliberately do
not use their COCO/VOC segmentation weights, because VOC contains "cat" and
"dog" classes and that would blur what the fine-tuning itself contributes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Large_Weights, ResNet18_Weights, resnet18
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, lraspp_mobilenet_v3_large


def conv_bn_relu(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(conv_bn_relu(cin, cout), conv_bn_relu(cout, cout))


class UNet(nn.Module):
    """U-Net with BatchNorm and padded convs (so output size == input size)."""

    def __init__(self, num_classes=2, base=32, depth=4):
        super().__init__()
        chs = [base * 2**i for i in range(depth + 1)]  # 16, 32, 64, 128, 256 for base=16
        self.downs = nn.ModuleList([DoubleConv(3, chs[0])])
        self.downs.extend(DoubleConv(chs[i], chs[i + 1]) for i in range(depth))
        self.ups = nn.ModuleList(
            nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in reversed(range(depth))
        )
        self.dec = nn.ModuleList(DoubleConv(chs[i] * 2, chs[i]) for i in reversed(range(depth)))
        self.head = nn.Conv2d(chs[0], num_classes, 1)

    def forward(self, x):
        skips = []
        for i, down in enumerate(self.downs):
            x = down(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.ups, self.dec):
            x = up(x)
            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:  # input not divisible by 2**depth
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([x, skip], dim=1))
        return self.head(x)


class DecoderBlock(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.conv = DoubleConv(cin + cskip, cout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNetResNet18(nn.Module):
    """U-Net whose encoder is a pretrained ResNet-18.

    Skip connections come from strides 2, 4, 8, 16; the bottleneck is layer4
    (stride 32). The final block upsamples back to the input resolution.
    """

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        r = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)  # /2, 64
        self.pool = r.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 32)
        self.head = nn.Sequential(conv_bn_relu(32, 16), nn.Conv2d(16, num_classes, 1))

    def forward(self, x):
        size = x.shape[-2:]
        s1 = self.stem(x)                    # /2
        s2 = self.layer1(self.pool(s1))      # /4
        s3 = self.layer2(s2)                 # /8
        s4 = self.layer3(s3)                 # /16
        x = self.layer4(s4)                  # /32
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.head(x)


class TorchvisionSeg(nn.Module):
    """Unwrap torchvision's {"out": logits} dict so all models share one API
    (a plain tensor output is also what ONNX export wants)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)["out"]


def build_model(name: str, num_classes=2, pretrained=True) -> nn.Module:
    bb = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
    if name == "unet_scratch":
        return UNet(num_classes, base=16)
    if name == "unet_resnet18":
        return UNetResNet18(num_classes, pretrained)
    if name == "lraspp_mbv3":
        return TorchvisionSeg(lraspp_mobilenet_v3_large(weights_backbone=bb, num_classes=num_classes))
    if name == "deeplabv3_mbv3":
        return TorchvisionSeg(deeplabv3_mobilenet_v3_large(
            weights_backbone=bb, num_classes=num_classes, aux_loss=False))
    raise ValueError(f"unknown model {name!r}")


MODELS = ["unet_scratch", "unet_resnet18", "lraspp_mbv3", "deeplabv3_mbv3"]
