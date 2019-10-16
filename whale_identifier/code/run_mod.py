import torch
from torch.autograd import Variable
from whale_identifier.code.models.simple_cnn import BasicCnn
from whale_identifier.code.data.datasets import WhaleDataset
from whale_identifier.code.data.transformations import ToTensor, Rescale
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import torch.nn as nn


CSV_PATH = "../input/labels/train.csv"
IMG_PATH = "../input/train/"
whale_data = WhaleDataset(CSV_PATH, IMG_PATH, transform=transforms.Compose(
    [Rescale((256,256)),ToTensor()]))


data_loader = DataLoader(whale_data, 1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BasicCnn(device)

CELoss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=.001)

for i, sample_data in enumerate(data_loader, 0):
    print(f'iteration {i}')
    batch = sample_data['image'].to(torch.float32)
    label = sample_data['label']
    batch = Variable(batch)
    optimizer.zero_grad()
    print(batch.type())
    output = model(batch)
    loss = CELoss(output, label)
    loss.backward()
    optimizer.step()
    print(f'loss {loss}')




