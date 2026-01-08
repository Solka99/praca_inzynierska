import torch
from torch import nn
import torch.nn.functional as F
import torch

# Wklejasz tu swoją klasę CNN_1
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        return self.block(x)

class CNN_Improved(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 64x64 -> 32x32
        self.block1 = ConvBlock(1, 32)
        # 32x32 -> 16x16
        self.block2 = ConvBlock(32, 64)
        # 16x16 -> 8x8
        self.block3 = ConvBlock(64, 128)

        self.gap  = nn.AdaptiveAvgPool2d(1)   # 8x8 -> 1x1
        self.drop = nn.Dropout(0.3)
        self.fc   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)          # [B, 128, 8, 8]
        x = self.gap(x)             # [B, 128, 1, 1]
        x = torch.flatten(x, 1)     # [B, 128]
        x = self.drop(x)
        x = self.fc(x)
        return x

# Twój model "kanji"
num_classes = 956
model = CNN_Improved(num_classes=num_classes)

# Sztuczne wejście 1x1x64x64 (tak jak opisałeś w komentarzach)
x = torch.randn(1, 1, 64, 64)
y = model(x)

# dot = make_dot(y, params=dict(model.named_parameters()))
# dot.render("cnn_kanji_architecture", format="png")  # zapisze png obok skryptu
#
# writer = SummaryWriter("runs/cnn_kanji")
# writer.add_graph(model, x)
# writer.close()


# model = CNN_1(num_classes=956)
# summary(model, input_size=(1, 1, 64, 64))
model = CNN_Improved(num_classes=956)
dummy = torch.randn(1, 1, 64, 64)

torch.onnx.export(
    model,
    dummy,
    "kanji_cnn.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)
summary(model, (1, 64, 64))