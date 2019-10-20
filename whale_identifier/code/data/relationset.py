import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io, transform
from torchvision import transforms, utils
import os

IMAGE_PATH = '../input/train/'
CSV_PATH = '../input/labels/train.csv'

class WhaleDataset(Dataset):
    """
    Whale Pytorch Dataset Class
    """

    def __init__(self, csv_file, img_dir, k_shot, transform=None):
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
        self.k_shot = k_shot
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
        not_new  = self.df.loc[self.df['Id'] != 'new_whale']
        image_counts = not_new.groupby(['Id']).count()
        greater_than1 = image_counts.index[image_counts['Image'] != 1]
        labeled_df = not_new.loc[not_new['Id'].isin(greater_than1)]
        return labeled_df

    def _build_label_dict(self):
        sorted_labels = sorted(self.df.Id.unique())
        label_to_number = {}
        number_to_label = {}
        for i, j in enumerate(sorted_labels):
            label_to_number[j] = i
            number_to_label[i] = j
        return label_to_number, number_to_label

    def _return_random_sample(self, idx):
        """
        provided a class, returns a random
        pair of images from the same class
        """
        label = self.labeled_df.iloc[idx, 1]
        image = self.labeled_df.iloc[idx, 0]
        class_subset = self.labeled_df.loc[self.labeled_df['Id'] == label]
        label_excluded = class_subset.loc[class_subset['Image'] != image]
        image_match = label_excluded.sample(n=self.k_shot)
        return image_match

    def __len__(self):
        """
        just the size of the dataset
        """
        return len(self.labeled_df)

    def __getitem__(self, idx):
        """
        return an example set of C classes from the
        data set (dictionary)
        """
        img_name = os.path.join(self.img_dir, self.labeled_df.iloc[idx, 0])
        image = io.imread(img_name)
        label_as_str = self.labeled_df.iloc[idx, 1]
        label = self.label_to_number[label_as_str]
        random_match = self._return_random_sample(idx)
        match_name = os.path.join(self.img_dir, random_match.iloc[0, 0])
        match = io.imread(match_name)
        match_label_str = random_match.iloc[0,1]
        match_label = self.label_to_number[match_label_str]
        sample = {'image' : image, 'label': label}
        matched_sample = {'image': match, 'label': match_label}
        if self.transform:
            sample = self.transform(sample)
            matched_sample = self.transform(matched_sample)

        return sample, matched_sample



