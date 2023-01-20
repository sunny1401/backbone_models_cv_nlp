from abc import abstractmethod
from dataclasses import asdict
import os
from typing import Callable, Optional

import pandas as pd
from torch.utils.data import Dataset
from torchvision import datasets


from src.cv.pytorch.datasets.configs import (
    CustomDatasetConfig,
    PytorchDatasetConfig, 
    PytorchAvailableDatasets
)

from src import CURRENT_ROOT_DIR

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class PyTorchDataset(Dataset):

    def __init__(
        self, 
        dataset_name: str,
        img_dir: str,
        composed_transforms: Callable
    ) -> None:
        super().__init__()


        dataset_location = os.path.join(CURRENT_ROOT_DIR, img_dir)
        os.makedirs(img_dir, exist_ok=True)

        pytorch_lib_callable: Callable = getattr(
            datasets, asdict(PytorchAvailableDatasets())[dataset_name])

        self.dataset = PytorchDatasetConfig(
            dataset_name=dataset_name, 
            dataset_download_location=dataset_location,
            test_dataset=pytorch_lib_callable(
                root=dataset_location, train=False, download=True, transform=composed_transforms
                ),
            train_dataset=pytorch_lib_callable(
                root=dataset_location, train=True, download=True, transform=composed_transforms
                ),
            
            )


class CustomDataset(Dataset):

    """
    Supports loading of local data for models. 
    In this repo, local data is generally downloaded via kaggle
    """
    def __init__(self,
        dataset_name: str,
        data_type: str,
        annotation_file: Optional[str] = None,
        composed_transforms: Callable = None,
        img_dir: Optional[str] = None,
        data_file: Optional[str] = None,
    ) -> None:

        super().__init__()
        """
        
        """

        if data_type not in {"image", "csv"}:
            raise ValueError(
                "System accepts either image_dir and annotation_file"
                " or train and test data files."
            )

        self._data_type = data_type

        img_dir_data_condition = (
            (annotation_file and img_dir)
            and (not data_file)
        )
        train_test_data = (
            (not annotation_file and not img_dir)
            and (data_file)
        )
        if img_dir_data_condition:
            img_dir = os.path.join(CURRENT_ROOT_DIR, img_dir)

            if not os.path.exists(img_dir):
                raise FileNotFoundError(f"Image directory for {dataset_name} not found at {img_dir}")

            if not annotation_file:
                raise ValueError("Please provide a valid input labels file")

            annotation_file = os.path.join(CURRENT_ROOT_DIR, annotation_file)
            if not os.path.exists(annotation_file):
                raise FileNotFoundError(f"Could not fild a valid annotation file for the image dataset {dataset_name}")

            
            self.dataset = CustomDatasetConfig(dataset_name=dataset_name, img_directory=img_dir, image_labels=pd.read_csv(annotation_file))
        
        elif train_test_data:

            data_file = os.path.join(CURRENT_ROOT_DIR, data_file)
            if not os.path.exists(data_file):
                raise FileNotFoundError("Training Data file not found. Please provide a valid file")


            self._load_custom_image_data(
                data_file=data_file,
                dataset_name=dataset_name
            )

        
        else:
            raise ValueError(
                "No useful data provided to the model."
                "Data needs to be in combination of training and"
                " test data or image_directory and annotation_file."
            )

        self.transform = composed_transforms

    @abstractmethod
    def _load_custom_image_data(self, data_file, dataset_name):
        pass

    def __len__(self):
        return self.dataset.image_labels.shape[0]

    @abstractmethod
    def __getitem__(self, idx):
        pass
