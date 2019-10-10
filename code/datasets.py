import pandas as pd
import torch
from torch.utils.data import Dataset

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


    def _remove_unkowns(self):
        """
        remove ids that are labeled
        as 'new whale
        """
        pass

    def __len__(self):
        """
        just the size of the dataset
        """
        pass

    def __getitem__(self):
        """
        return an example from the
        data set (dictionary)
        """
        pass
