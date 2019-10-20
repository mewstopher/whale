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

# training loop
count = 0
for data in data_loader:
    count +=1
    first_set = data[0]
    second_set = data[1]
    images1 = first_set['image']
    images2 = second_set['image']
    labels1 = first_set['label']
    labels2 = second_set['label']

    # send labels and images to device
    images1 = images1.to(device)
    images2 = images2.to(device)
    labels1 = labels1.to(device)
    labels2 = labels2.to(device)

    images1 = images1.to(torch.float)
    images2 = images2.to(torch.float)
    encoder_output = encoder(images1)
    batch_output = encoder(images2)
    #encoder_output_ext = encoder_output.unsqueeze(0).repeat(1*5,1,1,1,1)
    #batch_features_ext = batch_output.unsqueeze(0).repeat(1*5,1,1,1,1)
    #batch_features_ext_trans = torch.transpose(batch_features_ext,0,1)
    relation_pairs = torch.cat((encoder_output, batch_output),2).view(-1,64*2,62,62)

    relater_output = relater(relation_pairs)

    one_hot_labels = torch.nn.functional.one_hot(labels1.view(-1,1)).view(5,max(labels1).item()+1).to(torch.float)

    loss = MSELoss(relater_output, one_hot_labels)
    encoder.zero_grad()
    relater.zero_grad()
    loss.backward()
    encoder_optim.step()
    relater_optim.step()
    print(f'loss at iteration{count}: {loss}')
    if count ==20:
        break


one_hot_labels1 = Variable(torch.zeros(5, 5).scatter_(1, labelsll.view(-1,1), 1))
n= 5
indices = torch.randint(0,n, size=(5,1))
one_hot = torch.nn.functional.one_hot(indices).to(torch.float)
one_hot_one = one_hot.view(-1,5,5)

real_onehot = torch.nn.functional.one_hot(labels1.view(-1,1)).to(torch.float)
loss = MSELoss(relater_output, one_hot)
loss_2 = MSELoss(relater_output, one_hot_one)
