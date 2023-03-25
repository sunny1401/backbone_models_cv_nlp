import os
from abc import abstractmethod
from typing import Union, Tuple, Callable, Optional, List

import numpy as np

from torch.utils.data import Dataset


from src import CURRENT_ROOT_DIR

class BDDBaseDataset(Dataset):
    def __init__(self, root_dir: str, split: str, transform: Optional[Callable] = None):
        if split not in {"train", "val"}:
            raise RuntimeError (
                "provided split not in valid splits"
            )
        self._split = split
        self._root_dir = os.path.join(CURRENT_ROOT_DIR, root_dir)
        self._labels = self._get_labels()
        self._images = self._get_images()
        self.transform = transform

    @abstractmethod
    def _get_images(self):
        raise NotImplementedError

    @abstractmethod
    def _get_image_and_label(self, idx) -> Tuple(str, Union[np.array, str]):
        raise NotImplementedError
    
    @abstractmethod
    def _get_labels(self) -> Union[np.array, List]:
        raise NotImplementedError
    
    @abstractmethod
    def _get_length(self) -> int:
        raise NotImplementedError
    
    def __len__(self):
        return self._get_length()

    def __getitem__(self, index):
        image, label = self._get_image_and_label(idx=index)
        if self.transform is not None:
            image = self.transform(image)
        return image, label