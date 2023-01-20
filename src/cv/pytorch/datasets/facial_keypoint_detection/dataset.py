import os
import torch
import numpy as np
import pandas as pd

from skimage import io
from matplotlib import pyplot as plt
from torchvision import transforms
from src.cv.pytorch.datasets.base_dataset import CustomDataset
from typing import Optional
from src.cv.pytorch.datasets.configs import (
    CustomDatasetConfig,
)
from src.cv.pytorch.datasets.facial_keypoint_detection.transforms import (
    Grayscale,
    RandomCrop,
    Resize, 
    ToTensor
)
torch.backends.cudnn.deterministic = True


class FacialKeypointDataset(CustomDataset):

    def __init__(
        self, 
        data_type: str,
        dataset_name: str, 
        annotation_file: Optional[str] = None, 
        data_file: Optional[str] = None,
        image_column: Optional[str] = None,
        img_dir: Optional[str] = None, 
        random_crop_size: int = 224,
        resize_size: int = 256
    ) -> None:
        
        self.__image_column = image_column
        if data_type == "image":
            super().__init__(
                data_type = data_type,
                dataset_name=dataset_name, 
                img_dir=img_dir, 
                annotation_file=annotation_file, 
                composed_transforms=transforms.Compose([
                    Grayscale(data_type=data_type),
                    Resize(data_type=data_type, size=resize_size),
                    RandomCrop(random_crop_size),
                    ToTensor(),
                ])
            )
        
        else:

            super().__init__(
                data_type = data_type,
                dataset_name=dataset_name,
                data_file=data_file,
                composed_transforms=transforms.Compose([
                    Resize(data_type=data_type, size=resize_size),
                    Grayscale(data_type=data_type)
                ])
            )

    def _load_custom_image_data(self, data_file, dataset_name):
        
        data = pd.read_csv(data_file)
        if not self.__image_column:
            if "Image" not in data.columns:
                raise ValueError(
                    "No way to identify column containing image data"
                    "Please provide value in image_column argument "
                    "or put image data is 'Image' column"
                )
            self.__image_column = "Image"


        if not isinstance(data.loc[0, self.__image_column], np.ndarray):
            data[self.__image_column] = data[self.__image_column].apply(
                lambda x: np.array(x.split(" "), dtype="float"))
        data.fillna(method = 'ffill',inplace = True)
        image_data = data.pop(self.__image_column)
        self._image_labels = data

        self.dataset = CustomDatasetConfig(
            dataset_name=dataset_name, 
            image_labels=data, 
            image_data=image_data
        )

    def _get_image_data(self, idx):

        # getting image name from keypoint file
        img_name = os.path.join(self.dataset.img_directory, self.dataset.image_labels.iloc[idx, 0])
        # reading input file in
        image = io.imread(img_name)
        
        # extracting keypoints for the image
        image_label = np.array(
            self.dataset.image_labels.iloc[idx, 1:]).astype("float").reshape(-1, 2)
        sample = {'image': image, 'facial_landmarks': image_label}

        return sample

    def __getitem__(self, idx):
        """ 
        Function copied from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        
        """
        # get idx to be returned as a list
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self._data_type == "image":
            sample = self._get_image_data(self, idx)

        else:
            image = self.dataset.image_data[idx]
            keypoints = np.asarray(
                self.dataset.image_labels.iloc[idx, :]).astype('float').reshape(-1, 2)

            sample = {"image": image, "facial_landmarks": keypoints}

        return self.transform(sample)


    