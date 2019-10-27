import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io, transform
from torchvision import transforms, utils
import os

IMAGE_PATH = '../input/train/'
CSV_PATH = '../input/labels/train.csv'

class WhaleRelationset(Dataset):
    """
    Whale Pytorch Dataset Class
    """

    def __init__(self, csv_file, img_dir, k_way, transform=None):
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
        self.k_way = k_way
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
        image_match = label_excluded.sample(n=self.k_way, replace=True)
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
        match_names = [os.path.join(self.img_dir, random_match.iloc[i, 0]) for i in range(len(random_match))]
        matches = [io.imread(i) for i in match_names]
        match_label_str = random_match.iloc[0,1]
        match_label = self.label_to_number[match_label_str]

        sample = {'image' : image, 'label': label}
        matched_sample = {'image': matches, 'label': np.repeat(match_label, self.k_way)}
        if self.transform:
            sample = self.transform(sample)
            transformed_match_imgs= []
            for i in matched_sample['image']:
                match_dict = {'image': i, 'label': match_label}
                transformed_dict = self.transform(match_dict)
                transformed_match_imgs.append(transformed_dict['image'])
            matched_sample = {'image': torch.stack(transformed_match_imgs),
                              'label': np.repeat(match_label, self.k_way)}

        return sample, matched_sample


class Processor:

    def __init__(self, data, num_batches, classes, img_size, device):
        self.data = data
        self.nb = num_batches
        self.classes = classes
        self.img_sz = img_size
        self.device = device
        self.sample, self.sample_labels, self.batch, self.batch_labels = self.extract_data()
        if hasattr(self,'labels_to_indices') and hasattr(self, 'indices_to_label'):
            print('already has been instatntiated')
        else:
            self.labels_to_indices, self.indices_to_labels = self._build_dicts()

    def extract_data(self):
        sample_data, batch_data = self.data

        #extract data and send to device
        sample = sample_data['image']
        sample_labels = sample_data['label']
        batch = batch_data['image']
        batch_labels = batch_data['label']

        return sample, sample_labels, batch, batch_labels


    def process_sample(self):
        sample_labels = self._sample_to_indices()
        sample_labels = sample_labels.to(self.device)
        sample = self.sample.to(self.device)
        sample = sample.to(torch.float)
        return sample, sample_labels

    def process_batch(self):
        batch_labels = self._batch_to_indices()
        batch = self.batch.view(-1, self.nb*self.classes, 3, self.img_sz, self.img_sz).squeeze()
        batch, batch_labels = self.shuffle_batches(batch, batch_labels)
        batch = batch.to(self.device)
        batch_labels = batch_labels.to(self.device)
        batch = batch.to(torch.float)
        return batch, batch_labels

    def _build_dicts(self):
        """
        convert label(tensor) to its sorted index
        """
        labels_to_indices = {}
        indices_to_label = {}
        for i, j in enumerate(self.sample_labels):
            labels_to_indices[j.item()] = i
            indices_to_label[i] = j.item()

        return labels_to_indices, indices_to_label

    def _sample_to_indices(self):
        sample_labels = self.sample_labels
        for i, j in enumerate(self.sample_labels):
            sample_labels[i] = self.labels_to_indices[j.item()]

        return sample_labels

    def _batch_to_indices(self):
        """
        returns labels (sample or batch) based on
        index dict
        """
        batch_labels = self.batch_labels.view(-1)
        for i, j in enumerate(batch_labels):
            batch_labels[i] = self.labels_to_indices[j.item()]

        return  batch_labels


    def shuffle_batches(self, batch, batch_labels):
        """
        returns a randomly shuffle batches and labels
        arrays
        """
        indices = np.arange(batch.shape[0])
        np.random.shuffle(indices)
        batch_labels = batch_labels[indices]
        batch = batch[indices]

        return batch, batch_labels


