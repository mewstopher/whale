import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F
from whale_identifier.code.models.simple_cnn import BasicCnn
from whale_identifier.code.data.datasets import WhaleDataset
from whale_identifier.code.data.transformations import ToTensor, Rescale
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import logging

LOG = logging.getLogger(__name__)

CSV_PATH = "../input/labels/train.csv"
IMG_PATH = "../input/train/"
whale_data = WhaleDataset(CSV_PATH, IMG_PATH, transform=transforms.Compose(
    [Rescale((256,256)),ToTensor()]))


data_loader = DataLoader(whale_data, 1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# trying pre-built vgg(no pre-trained)
class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

vgg = models.vgg16()
model = nn.Sequential(*list(vgg.children())[:2])
model.classifier = nn.Sequential(
    Flatten(),
    nn.Linear(25088, 10000),
    nn.ReLU(),
    nn.Linear(10000, 5004)
)
model = BasicCnn(device)

CELoss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=.001)

for i, sample_data in enumerate(data_loader, 0):
    batch = sample_data['image'].to(torch.float32)
    label = sample_data['label']
    batch = Variable(batch)
    optimizer.zero_grad()
    output = model(batch)
    loss = CELoss(output, label)
    loss.backward()
    optimizer.step()
    LOG.info(loss)
#    print(f'loss {loss}')



import torchvision.models as models
vgg = models.vgg16()
newmod = nn.Sequential(*list(vgg.children())[:2])
newmod.classifier = nn.Sequential(
    nn.ReLU(nn.Linear(25088, 10000)),
    nn.Linear(10000, 5004)
)

