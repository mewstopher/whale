import torch
from whale_identifier.code.models.relationet import CNNEncoder, RelationNetwork
from whale_identifier.code.data.relationset import WhaleRelationset, Processor
from whale_identifier.code.data.transformations import ToTensor, Rescale, Normalize
from whale_identifier.code.trainers.train_relationet import TrainRelationet
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import logging
from whale_identifier.code.trainers.train_mod import WhaleTrainer

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


trainer = TrainRelationet(encoder, relater, MSELoss, encoder_optim, relater_optim,
                          data_loader, 1, 19, 5,IMG_SIZE, device)
trainer.run()

# training loop
count = 0
#for data in data_loader:

# use next iter() for testing
data = next(iter(data_loader))
count +=1
sample, batch = data

#extract data and send to device
sample, sample_labels = sample['image'].to(device), sample['label'].to(device)
batch, batch_labels = batch['image'].to(device), batch['label'].to(device)
batch = batch.view(-1, NUM_BATCHES*CLASSES, 3, IMG_SIZE, IMG_SIZE).squeeze()

#change batch_labels to a 1d  Tensor
batch_labels = batch_labels.view(-1)

sample_labels, batch_labels = whale_data.convert_to_indices(sample_labels, batch_labels)
batch, batch_labels = whale_data.shuffle_batches(batch, batch_labels)
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



