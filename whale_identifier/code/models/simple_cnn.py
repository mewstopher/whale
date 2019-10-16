import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicCnn(nn.Module):
    """
    basic cnn
    """
    def __init__(self, device):
        super(BasicCnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 3)
        self.conv2 = nn.Conv2d(10,64, 3)
        self.conv3 = nn.Conv2d(64,128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.conv5 = nn.Conv2d(256, 512, 3)
        self.fc1 = nn.Linear(18432, 10000)
        self.fc2 = nn.Linear(10000, 5004)
        self.to(device)

    def forward(self, x):
        X = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        X = F.max_pool2d(F.relu(self.conv2(X)), (2,2))
        X = F.max_pool2d(F.relu(self.conv3(X)), (2,2))
        X = F.max_pool2d(F.relu(self.conv4(X)), (2,2))
        X = F.max_pool2d(F.relu(self.conv5(X)), (2,2))
        X = X.view(-1, self.num_flat_features(X))
        X = F.relu(self.fc1(X))
        X = self.fc2(X)

        return X

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
