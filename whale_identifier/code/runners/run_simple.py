import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F
from whale_identifier.code.models.simple_cnn import BasicCnn
from whale_identifier.code.data.datasets import WhaleDataset
from whale_identifier.code.data.transformations import ToTensor, Rescale, Normalize
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import logging
from whale_identifier.code.trainers.train_mod import WhaleTrainer
LOG = logging.getLogger(__name__)

CSV_PATH = "../input/labels/train.csv"
IMG_PATH = "../input/train/"
whale_data = WhaleDataset(CSV_PATH, IMG_PATH, transform=transforms.Compose(
    [Rescale((256,256)),ToTensor()]))


data_loader = DataLoader(whale_data, 1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = BasicCnn(device)

CELoss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=.0001)

num_epochs = 1

trainer = WhaleTrainer(model, data_loader, data_loader, num_epochs, CELoss, optimizer)
trainer.train()


