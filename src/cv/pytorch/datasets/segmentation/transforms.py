from torchvision import transforms
from typing import Dict, Union, Tuple


class ToTensor:
    def __call__(self, sample: Dict) -> Dict:

        image = sample["image"]
        labels = sample.get("label", None)
         
        to_tensor_transform = transforms.ToTensor()

        return {
            'image': to_tensor_transform(image),
            'label': to_tensor_transform(labels)
        }


class RandomCrop:
    
    def __init__(self, size: Union[int, Tuple[int, int]]) -> None:
        if not isinstance(size, (int, tuple)):
            raise ("Argument Size takes values of type int or tuple")
            
        self._size = size
            
    def __call__(self, sample: Dict) -> Dict:

        image = sample["image"]
        labels = sample.get("label", None)
        random_crop_transform = transforms.RandomCrop(self._size)
        
        return {
            'image': random_crop_transform(image),
            'label': random_crop_transform(labels)
        }


class Resize:

    def __init__(self, size: Union[int, Tuple[int, int]]) -> None:
        if not isinstance(size, (int, tuple)):
            raise ("Argument Size takes values of type int or tuple")
            
        self._size = size

    def __call__(self, sample: Dict) -> Dict:

        image = sample["image"]
        labels = sample.get("label", None)
        random_crop_transform = transforms.Resize(self._size)
        
        return {
            'image': random_crop_transform(image),
            'label': random_crop_transform(labels)
        }


class GaussianBlur:

    def __init__(self, size: Union[int, Tuple[int, int]]) -> None:
        if not isinstance(size, (int, tuple)):
            raise ("Argument Size takes values of type int or tuple")
            
        self._size = size

    def __call__(self, sample: Dict) -> Dict:
        image = sample["image"]
        labels = sample.get("label", None)

        gaussian_blur = transforms.GaussianBlur(self._size)

        return {
            'image': gaussian_blur(image),
            'label': gaussian_blur(labels)
        }