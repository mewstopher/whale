import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import time 
import tqdm
from PIL import Image
train_on_gpu = True
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

from collections import OrderedDict
import cv2



import albumentations
from albumentations import torch as AT
import pretrainedmodels




# Transformations
# Basic transformations include only resizing the image to the necessary size, converting to Pytorch tensor and normalizing

data_transforms = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
])

data_transforms_test = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
])



# Encoding labels
# Labels need to be one-hot encoded.

y, le = prepare_labels(train_df['Id'])


# Dataset
# Now we need to create a dataset. Sadly, default version won't work, as images for each class are supposed to be in separate folders. So I write a custom WhaleDataset.

class WhaleDataset(Dataset):
    def __init__(self, datafolder, datatype='train', df=None, transform = transforms.Compose([transforms.ToTensor()]), y=None):
        self.datafolder = datafolder
        self.datatype = datatype
        self.y = y
        if self.datatype == 'train':
        self.df = df.values
        self.image_files_list = [s for s in os.listdir(datafolder)]
        self.transform = transform


    def __len__(self):
        return len(self.image_files_list)

    def __getitem__(self, idx):
        if self.datatype == 'train':
        img_name = os.path.join(self.datafolder, self.df[idx][0])
        label = self.y[idx]

        elif self.datatype == 'test':
        img_name = os.path.join(self.datafolder, self.image_files_list[idx])
        label = np.zeros((5005,))

        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)
        if self.datatype == 'train':
        return image, label
        elif self.datatype == 'test':
        # so that the images will be in a correct order
        return image, label, self.image_files_list[idx]

train_dataset = WhaleDataset(datafolder='../input/train/', datatype='train', df=train_df, transform=data_transforms, y=y)
test_set = WhaleDataset(datafolder='../input/test/', datatype='test', transform=data_transforms_test)


# Loaders
# Now we create loaders. Here we define which images will be used, batch size and other things.

train_sampler = SubsetRandomSampler(list(range(len(os.listdir('../input/train')))))
valid_sampler = SubsetRandomSampler(list(range(len(os.listdir('../input/test')))))
batch_size = 512
num_workers = 0

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
# less size for test loader.
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, num_workers=num_workers)


# Basic CNN
# Now we can define the model. For now I'll use a simple architecture with two convolutional layers




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 7, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)        
        self.pool2 = nn.AvgPool2d(3, 3)

        self.fc1 = nn.Linear(64 * 4 * 4 * 16, 1024)
        self.fc2 = nn.Linear(1024, 5005)

        self.dropout = nn.Dropout(0.5)        

    def forward(self, x):
        x = self.pool(F.relu(self.conv2_bn(self.conv1(x))))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# Initializing model
# We need to define model, loss, oprimizer and possibly a scheduler.

model_conv = Net()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model_conv.parameters(), lr=0.01)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)




# model_conv.cuda()
# n_epochs = 10
# for epoch in range(1, n_epochs+1):
#     print(time.ctime(), 'Epoch:', epoch)

#     train_loss = []
#     exp_lr_scheduler.step()

#     for batch_i, (data, target) in enumerate(train_loader):
#         #print(batch_i)
#         data, target = data.cuda(), target.cuda()

#         optimizer.zero_grad()
#         output = model_conv(data)
#         loss = criterion(output, target.float())
#         train_loss.append(loss.item())

#         loss.backward()
#         optimizer.step()

#     print(f'Epoch {epoch}, train loss: {np.mean(train_loss):.4f}')

# sub = pd.read_csv('../input/sample_submission.csv')

# model_conv.eval()
# for (data, target, name) in test_loader:
#     data = data.cuda()
#     output = model_conv(data)
#     output = output.cpu().detach().numpy()
#     for i, (e, n) in enumerate(list(zip(output, name))):
#         sub.loc[sub['Image'] == n, 'Id'] = ' '.join(le.inverse_transform(e.argsort()[-5:][::-1]))
        
        # sub.to_csv('basic_model.csv', index=False)

