import os
from typing import Tuple, Callable, Optional

import numpy as np
from PIL import Image

from torch.utils.data import Dataset
import torch

from src import CURRENT_ROOT_DIR


class WeatherDataset(Dataset):

    WEATHER_SET = dict(Cloudy=1, Foggy=2, Rainy=3, Snowy=4, Sunny=5)
    def __init__(self, root_dir: str, weather_type: str, transform: Optional[Callable] = None):

        if weather_type not in self.WEATHER_SET:
            raise RuntimeError(
                " Value provided for weather doesn't make any sense. "
                f"Please provide a value from {self.WEATHER_SET}"
            )

        self.transform = transform
        root_dir = os.path.join(CURRENT_ROOT_DIR, root_dir, weather_type)

        self._label = self.WEATHER_SET[weather_type]

        self._images = []

        for dirpath, _, filenames in os.walk(self._root_dir):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    self._images.append(os.path.join(dirpath, filename))
        
    def __len__(self) -> int:

        return len(self._images)
    
    def __getitem__(self, index) -> Tuple[np.array, int]:

        image_path = self._images[index]
        image =  Image.open(image_path)
        label = torch.tensor(self._label, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label
    
    def file_paths(self, idx):
        return self._images[idx]