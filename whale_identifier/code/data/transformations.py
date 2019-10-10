import torch
from torchvision import utils, transforms
from skimage import transform

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

        return {'image': image, 'label' : label}



