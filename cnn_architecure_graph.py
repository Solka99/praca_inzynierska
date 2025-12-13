import torch
from torch import nn
import torch.nn.functional as F
import torch

# Wklejasz tu swoją klasę CNN_1
class CNN_1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0, stride=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.drop  = nn.Dropout(0.2)
        self.fc    = nn.Linear(128 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.flatten(1)
        x = self.drop(x)
        return self.fc(x)

# Twój model "kanji"
num_classes = 956
model = CNN_1(num_classes=num_classes)

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
model = CNN_1(num_classes=956)
dummy = torch.randn(1, 1, 64, 64)

torch.onnx.export(
    model,
    dummy,
    "kanji_cnn.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)