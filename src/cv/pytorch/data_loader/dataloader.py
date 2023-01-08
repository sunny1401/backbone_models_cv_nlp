from abc import abstractmethod
from dataclasses import asdict
import os
from typing import Callable

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets


from .configs import (
    CURRENT_ROOT_DIR, 
    CustomDataset,
    PytorchDataset, 
    PytorchLibDatasets
)

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class DatasetLoader(Dataset):

    def __init__(
        self, 
        dataset_name: str,
        img_dir: str,
        composed_transforms: Callable
    ) -> None:
        super().__init__()


        dataset_location = os.path.join(CURRENT_ROOT_DIR, img_dir)
        os.makedirs(img_dir, exist_ok=True)

        pytorch_lib_callable: Callable = getattr(datasets, asdict(PytorchLibDatasets())[dataset_name])

        self.dataset = PytorchDataset(
            dataset_name=dataset_name, 
            dataset_download_location=dataset_location,
            test_dataset=pytorch_lib_callable(
                root=dataset_location, train=False, download=True, transform=composed_transforms
                ),
            train_dataset=pytorch_lib_callable(
                root=dataset_location, train=True, download=True, transform=composed_transforms
                ),
            
            )


class CustomDatasetLoader(Dataset):

    def __init__(self,
        dataset_name: str,
        img_dir: str,
        annotation_file: str = None,
        composed_transforms: Callable = None,
    ) -> None:
        # TODO - add code for download
        super().__init__()

        img_dir = os.path.join(CURRENT_ROOT_DIR, img_dir)

        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory for {dataset_name} not found at {img_dir}")

        if not annotation_file:
            raise ValueError("Please provide a valid input labels file")

        annotation_file = os.path.join(CURRENT_ROOT_DIR, annotation_file)
        if not os.path.exists(annotation_file):
            raise FileNotFoundError(f"Could not fild a valid annotation file for the image dataset {dataset_name}")

        
        self._image_labels = pd.read_csv(annotation_file)
        self.dataset = CustomDataset(dataset_name=dataset_name, img_directory=img_dir, image_labels=self._image_labels)

        self.transform = composed_transforms

    def __len__(self):
        return self._image_labels.shape[0]

    @abstractmethod
    def __getitem__(self, idx):
        pass
