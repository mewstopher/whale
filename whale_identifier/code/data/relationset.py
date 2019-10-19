import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io, transform
from torchvision import transforms, utils
#from .transformations import Rescale
import os

IMAGE_PATH = '../input/train/'
CSV_PATH = '../input/labels/train.csv'

class WhaleDataset(Dataset):
    """
    Whale Pytorch Dataset Class
    """

    def __init__(self, csv_file, img_dir, transform=None):
        """
        load images, labels
        get them into a dictionary of some sort

        PARAMS
        -------------------------
        csv_file: csv with the labels
        img_dir: path to folder containing images
        """
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.label_to_number, self.number_to_label = self._build_label_dict()
        self.labeled_df = self._remove_unkowns()


    def _remove_unkowns(self):
        """
        remove ids that are labeled
        as 'new whale
        Also remove data where the there
        is only one example fo a class
        """
        labeled_df = self.df.loc[self.df['Id'] != 'new_whale']
        return labeled_df

    def _build_label_dict(self):
        sorted_labels = sorted(self.df.Id.unique())
        label_to_number = {}
        number_to_label = {}
        for i, j in enumerate(sorted_labels):
            label_to_number[j] = i
            number_to_label[i] = j
        return label_to_number, number_to_label

    def __len__(self):
        """
        just the size of the dataset
        """
        return len(self.labeled_df)

    def __getitem__(self, idx):
        """
        return an example from the
        data set (dictionary)
        """
        img_name = os.path.join(self.img_dir, self.labeled_df.iloc[idx, 0])
        image = io.imread(img_name)
        label_as_str = self.labeled_df.iloc[idx, 1]
        label = self.label_to_number[label_as_str]
        sample = {'image' : image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample



