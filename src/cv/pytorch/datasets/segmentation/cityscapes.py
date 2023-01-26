import torch
from src.cv.pytorch.datasets.base_dataset import PyTorchDataset
from typing import Callable, Optional
import torch


torch.backends.cudnn.deterministic = True

class CityScapeSegmentationDataset(PyTorchDataset):
    """
    Dataset currently downloaded from 
    https://www.kaggle.com/datasets/xiaose/cityscapes
    Assumes both gtFine and leftImg8bit are available for the split_type.
    Will require gtCoarse for data_type = coarse
    """
    
    def __init__(
        self, 
        dataset_name: str,
        img_dir: str,
        data_type: str = "fine", 
        transforms: Optional[Callable] = None,
        split_type: str = "train"
    ):
        self._data_type = data_type
        self.transforms = transforms
        super().__init__(
            dataset_name=dataset_name,
            img_dir= img_dir,
            split_type=split_type
        )
        self._number_of_classes = len(self.dataset.classes)
        self._classes = self.dataset.classes
        self.transform = transforms

        
    def _initialize_data(
        self, dataset_location, pytorch_callable, split_type
    ):
        
        return pytorch_callable(
            root=dataset_location, 
            split=split_type, 
            target_type="semantic", 
            mode=self._data_type
        )
        
    def __getitem__(self, idx):
        
        # get idx to be returned as a list
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # reading input file in
        image, semantic_label = self.dataset[idx]
        
        sample = dict(image=image, label=semantic_label)
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def __len__(self):
        
        return len(self.dataset)