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
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# set PATHs
CSV_PATH = "../input/labels/train.csv"
IMG_PATH = "../input/train/"

NUM_BATCHES = 19
CLASSES = 5
IMG_SIZE = 256

# create data set with WaleRelationset class
whale_data = WhaleRelationset(CSV_PATH, IMG_PATH, 19, transform=transforms.Compose(
    [Rescale((256,256)),ToTensor(),Normalize()]))


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



# training loop
count = 0
for data in data_loader:
    count +=1
    sample, batch = data
    sample, sample_labels = sample['image'], sample['label']
    batch, batch_labels = batch['image'], batch['label']
    batch = batch.view(-1, NUM_BATCHES*CLASSES, 3, IMG_SIZE, IMG_SIZE).squeeze()

    # send labels and images to device
    # add here
    sample = sample.to(torch.float)
    batch = batch.to(torch.float)
    encoder_output = encoder(sample)
    batch_output = encoder(batch)
    encoder_output_ext = encoder_output.unsqueeze(0).repeat(NUM_BATCHES*CLASSES,1,1,1,1)
    batch_features_ext = batch_output.unsqueeze(0).repeat(1*CLASSES,1,1,1,1)
    batch_features_ext_trans = torch.transpose(batch_features_ext,0,1)
    #relation_pairs = torch.cat((encoder_output, batch_output),2).view(-1,64*2,62,62)
    relation_pairs = torch.cat((encoder_output_ext, batch_features_ext_trans),2).view(-1,64*2,62,62)

    relater_output = relater(relation_pairs).view(-1, CLASSES)
    one_hot_labels = Variable(torch.zeros(NUM_BATCHES*CLASSES, CLASSES).scatter_(1, batch_labels.view(-1, 1), 1))
    loss = MSELoss(relater_output, one_hot_labels)
    encoder.zero_grad()
    relater.zero_grad()
    loss.backward()
    encoder_optim.step()
    relater_optim.step()
    print(f'loss at iteration{count}: {loss}')
    if count ==200:
        break


one_hot_labels1 = Variable(torch.zeros(5, 5).scatter_(1, labelsll.view(-1,1), 1))
n= 5
indices = torch.randint(0,n, size=(5,1))
one_hot = torch.nn.functional.one_hot(indices).to(torch.float)
one_hot_one = one_hot.view(-1,5,5)

real_onehot = torch.nn.functional.one_hot(labels1.view(-1,1)).to(torch.float)
loss = MSELoss(relater_output, one_hot)
loss_2 = MSELoss(relater_output, one_hot_one)
