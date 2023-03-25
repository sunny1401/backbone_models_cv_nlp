import os
import numpy as np
from PIL import Image
from src.cv.pytorch.datasets.segmentation.bdd_base import BDDBaseDataset


class BDDSegmentationDataset(BDDBaseDataset):

    def __init__(
            self, 
            root_dir, 
            split='train', 
            transform=None
        ):

        root_dir = os.path.join(root_dir, split)
        super().__init__(root_dir, split, transform)


    def _get_images(self):
        images = []
        for dirpath, _, filenames in os.walk(self._root_dir):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    images.append(os.path.join(dirpath, filename))
        return images

    def _get_length(self) -> int:
        return len(self._images)

    def _get_labels(self):
        label_file = os.path.join(self._root_dir, f'{self._split}_labels.npy')
        return np.load(label_file)
    
    def _get_image_and_label(self, idx):
        image_path = self._images[idx]
        image = Image.open(image_path).convert('RGB')
        label = self._labels[idx]

        return image, label