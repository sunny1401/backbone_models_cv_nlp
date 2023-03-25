import os
from typing import Union

import pandas as pd
from PIL import Image
import torch
from src.dataset.bdd_base import BDDBaseDataset


class BDDWeatherDataset(BDDBaseDataset):
    def __init__(
            self, 
            root_dir, 
            weather_column: Union[str, int],
            csv_file: str ="labels.csv", 
            split ="train", 
            transform = None
        ):

        self.__csv_file = csv_file
        self.__weather_column = weather_column
        super().__init__(root_dir, split, transform)

    def _get_images(self):
        return pd.read_csv(self.__csv_file).iloc[:, 0].tolist()
 
    def _get_image_and_label(self, idx):
        img_name = os.path.join(self._root_dir, self._images[idx])
        image = Image.open(img_name)
        label = self._labels[idx]
        label = torch.tensor(label, dtype=torch.float32)
        return image, label

    
    def _get_labels(self):
        df = pd.read_csv(self.__csv_file)
        if isinstance(self.__weather_column, int):
            self.__weather_column = self.df.columns.tolist()[self.__weather_column]

        return df[self.__weather_column].tolist()

    def _get_length(self) -> int:
        return self._labels.shape[0]