import torch
from torchvision import utils, transforms
from skimage import transform
import numpy as np


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], np.array(sample['label'])
        if image.ndim != 3:
            image = np.stack((image,)*3, axis=-1)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),'label': torch.from_numpy(label)}


class Rescale(object):
    """
    Object to rescale images so that
    all of the images are the same size
    """

    def __init__(self, output_size):
        """
        PARAMS
        -------------
        output_size: desired size that
            the images should be rescaled to
        """
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        """
        takes in a sample of the data. Uses the
        output size to resize the original image
        so that all images are same size
        """
        image, label = sample['image'], sample['label']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h/w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w/h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'label' : label}



