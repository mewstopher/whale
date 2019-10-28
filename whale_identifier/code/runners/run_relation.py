import torch
from whale_identifier.code.models.relationet import CNNEncoder, RelationNetwork
from whale_identifier.code.data.relationset import WhaleRelationset, Processor
from whale_identifier.code.data.transformations import ToTensor, Rescale, Normalize
from whale_identifier.code.trainers.train_relationet import TrainRelationet
from whale_identifier.code.helper_functions import train_test_sampler
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import logging
import logging.config
from whale_identifier.code.trainers.train_mod import WhaleTrainer

logging.config.fileConfig('logging.ini')
# set PATHs
CSV_PATH = "../input/labels/train.csv"
IMG_PATH = "../input/train/"

NUM_BATCHES = 19
CLASSES = 3
IMG_SIZE = 256
EPOCHS = 1
# create data set with WaleRelationset class
whale_data = WhaleRelationset(CSV_PATH, IMG_PATH, 19, transform=transforms.Compose(
    [Rescale((256,256)),ToTensor(),Normalize()]))

# train, val test sets
train_sampler, val_sampler, test_sampler = train_test_sampler(whale_data, .8, .1, .1)

# get data into Dataloader
train_loader = DataLoader(whale_data, CLASSES, train_sampler)
val_loader = DataLoader(whale_data, CLASSES, val_sampler)
test_loader = DataLoader(whale_data, CLASSES, test_sampler)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize models
encoder  = CNNEncoder(device)
relater = RelationNetwork(64, 8, device)

# set up loss and optimizer
MSELoss = nn.MSELoss()
encoder_optim = torch.optim.Adam(encoder.parameters(), lr=.0001)
relater_optim = torch.optim.Adam(relater.parameters(), lr=.0001)


trainer = TrainRelationet(encoder, relater, MSELoss, encoder_optim, relater_optim,
                          train_loader,val_loader,EPOCHS, NUM_BATCHES, CLASSES,IMG_SIZE, device, show_every=1, save=True,
                          save_path = '../output/')

trainer.run()


