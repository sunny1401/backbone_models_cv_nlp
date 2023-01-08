from tokenize import Name
from typing import  NamedTuple
from torch.utils.data import Dataset
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

CURRENT_ROOT_DIR = str(Path(__file__).parent.parent.parent.parent)

@dataclass(frozen=True)
class PytorchLibDatasets:
    fashion_mnist: "FashionMNIST"
    mnist: "MNIST"


class PytorchDataset(NamedTuple):
    dataset_name: str
    train_dataset: Dataset
    test_datset: Dataset
    dataset_download_location: str


class CustomDataset(NamedTuple):
    dataset_name: str
    img_directory: str
    image_labels: pd.DataFrame