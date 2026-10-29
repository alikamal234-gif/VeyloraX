import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class SimpleYOLO(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.num_classes = num_classes
        out_c = 4 + 1 + num_classes

        self.c1 = ConvBlock(3, 32, stride=2)
        self.c2 = ConvBlock(32, 64, stride=2)
        self.c3 = ConvBlock(64, 128, stride=2)
        self.c4 = ConvBlock(128, 256, stride=2)
        self.c5 = ConvBlock(256, 512, stride=2)

        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        self.fpn_conv1 = ConvBlock(512 + 256, 256)
        self.fpn_conv2 = ConvBlock(256 + 128, 128)

        self.pan_down1 = ConvBlock(128, 256, stride=2)
        self.pan_conv1 = ConvBlock(256 + 256, 256)
        self.pan_down2 = ConvBlock(256, 512, stride=2)
        self.pan_conv2 = ConvBlock(512 + 512, 512)

        self.head_s = nn.Sequential(ConvBlock(128, 128), nn.Conv2d(128, out_c, 1))
        self.head_m = nn.Sequential(ConvBlock(256, 256), nn.Conv2d(256, out_c, 1))
        self.head_l = nn.Sequential(ConvBlock(512, 512), nn.Conv2d(512, out_c, 1))

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        c5 = self.c5(c4)

        n4 = self.fpn_conv1(torch.cat([self.up(c5), c4], dim=1))
        p3 = self.fpn_conv2(torch.cat([self.up(n4), c3], dim=1))

        p4 = self.pan_conv1(torch.cat([self.pan_down1(p3), n4], dim=1))
        p5 = self.pan_conv2(torch.cat([self.pan_down2(p4), c5], dim=1))

        out_s = self.head_s(p3)
        out_m = self.head_m(p4)
        out_l = self.head_l(p5)

        return out_s, out_m, out_l


import torch
import torch.nn.functional as F


def compute_loss(preds, targets, num_classes=21):
    device = preds[0].device
    total_loss = torch.tensor(0.0, device=device)

    for pred, S in zip(preds, [80, 40, 20]):
        B = pred.shape[0]
        pred = pred.permute(0, 2, 3, 1)

        target = torch.zeros_like(pred)
        mask = torch.zeros((B, S, S), dtype=torch.bool, device=device)

        for b, boxes in enumerate(targets):
            for cls, x, y, w, h in boxes:
                i, j = min(int(x*S), S-1), min(int(y*S), S-1)
                target[b, j, i, :5] = torch.tensor(
                    [x*S-i, y*S-j, w, h, 1.0], device=device
                )
                target[b, j, i, 5+int(cls)] = 1
                mask[b, j, i] = True

        loss_obj = F.binary_cross_entropy_with_logits(
            pred[..., 4], target[..., 4]
        )

        if mask.any():
            loss_box = F.mse_loss(pred[mask][..., :4], target[mask][..., :4])
            loss_cls = F.cross_entropy(
                pred[mask][..., 5:],
                target[mask][..., 5:].argmax(-1)
            )
            total_loss += 5 * loss_box + loss_obj + loss_cls
        else:
            total_loss += loss_obj

    return total_loss


def train_model(model, train_loader, val_loader=None, epochs=10, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = [t.to(device) for t in targets]

            optimizer.zero_grad()
            loss = compute_loss(model(images), targets)
            loss.backward()
            optimizer.step()
            total += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total/len(train_loader):.4f}")

    return model


if __name__ == "__main__":
    model = SimpleYOLO(num_classes=21)
    x = torch.randn(1, 3, 640, 640)
    out_s, out_m, out_l = model(x)
    print("Output shapes:", out_s.shape, out_m.shape, out_l.shape)
