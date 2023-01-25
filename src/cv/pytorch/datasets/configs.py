from typing import Optional
from typing import  NamedTuple
from torch.utils.data import Dataset
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class PytorchAvailableDatasets:
    fashion_mnist: str = "FashionMNIST"
    mnist: str = "MNIST"
    cityscapes: str = "Cityscapes"


class PytorchDatasetConfig(NamedTuple):
    dataset_name: str
    train_dataset: Dataset
    test_datset: Dataset
    dataset_download_location: str


class CustomDatasetConfig(NamedTuple):
    dataset_name: str
    image_labels: Optional[pd.DataFrame] = None
    img_directory: Optional[str] = None
    image_data: Optional[np.array] = None
