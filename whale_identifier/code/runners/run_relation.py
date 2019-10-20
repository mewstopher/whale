import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F
from whale_identifier.code.models.relationet import CNNEncoder, RelationNetwork
from whale_identifier.code.data.relationset import WhaleRelationset
from whale_identifier.code.data.transformations import ToTensor, Rescale, Normalize
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import logging
from whale_identifier.code.trainers.train_mod import WhaleTrainer
LOG = logging.getLogger(__name__)

# set PATHs
CSV_PATH = "../input/labels/train.csv"
IMG_PATH = "../input/train/"

# create data set with WaleRelationset class
whale_data = WhaleRelationset(CSV_PATH, IMG_PATH, 1, transform=transforms.Compose(
    [Rescale((256,256)),ToTensor()]))

# get data into Dataloader
data_loader = DataLoader(whale_data, 5)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize models
encoder  = CNNEncoder(device)
relater = RelationNetwork(64, 8, device)

# set up loss and optimizer
MSELoss = nn.MSELoss()
encoder_optim = torch.optim.Adam(encoder.parameters(), lr=.0001)
relater_optim = torch.optim.Adam(relater.parameters(), lr=.0001)

data = data_loader.__iter__().next()
images1 = data[0]['image']
images2 = data[1]['image']

for i, j in enumerate(images1):
    images1[i] = j.to(device)

images1 = Variable(data[0]['image'])
images1 = images1.to(torch.float)
images2 = images2.to(torch.float)
encoder_output = encoder(images1)
batch_output = encoder(images2)
encoder_output_ext = encoder_output.unsqueeze(0).repeat(1*5,1,1,1,1)
batch_features_ext = batch_output.unsqueeze(0).repeat(1*5,1,1,1,1)
batch_features_ext_trans = torch.transpose(batch_features_ext,0,1)
relation_pairs = torch.cat((encoder_output_ext, batch_features_ext_trans),2).view(-1,64*2,5,5)

relater_output = relater(relation_pairs)

labels1 = data[0]['label']
one_hot_labels = Variable(labels1.view(-1,1).to(torch.float))
loss = MSELoss(relater_output, one_hot_labels)

