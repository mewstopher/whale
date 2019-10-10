import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io, transform
from torchvision import transforms, utils
from .transformations import Rescale

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
        self.labeled_df = self._remove_unkowns()


    def _remove_unkowns(self):
        """
        remove ids that are labeled
        as 'new whale
        """
        labeled_df = self.df.loc[self.df['Id'] != 'new_whale']
        return labeled_df

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
        if torch.is_tensor(idx):
            ids = idx.to_list()
        img_name = os.path.join(self.img_dir, self.labeled_df[idx, 0])
        image = io.read(img_name)
        label = self.labeled_df.iloc[idx, 1]
        sample = {'image' : image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample



