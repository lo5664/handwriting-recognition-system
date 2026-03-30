import torch
import torch.nn as nn
import torch.nn.functional as F

class HandwritingCNN(nn.Module):
    def __init__(self):
        super(HandwritingCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 for 10 classes (digits 0-9)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class HandwritingRecognizer:
    def __init__(self):
        self.model = HandwritingCNN()
        self.model.load_state_dict(torch.load('handwriting_cnn.pth'))
        self.model.eval()

    def recognize(self, image_tensor):
        with torch.no_grad():
            output = self.model(image_tensor)
            _, predicted = torch.max(output, 1)
            return predicted.item()
