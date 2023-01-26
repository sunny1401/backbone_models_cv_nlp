from typing import Dict, Union, Tuple, Optional
import numpy as np
import cv2
import torch

class Grayscale:
    
    def __init__(self, data_type: str):

        if data_type not in {"image", "csv"}:
            raise ValueError(
                "Arg data_type must be an element of {image, csv}"
                f"You provided: {data_type}"
            )

        self._data_type = data_type

    def __call__(self, data: Dict):
        
        image = data["image"]
        key_points = data.get("facial_landmarks", None)
        if self._data_type == "image":
            
            # rescaling [0 to 1] from [0, 255]
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)/255
            return {'image': grayscale_image, "facial_landmarks": key_points}
        
        else:
            return {"image": image/255, "facial_landmarks": key_points}
    

class Resize:
    
    def __init__(self, data_type: str, size: Union[int, Tuple[int, int]]):

        
        if data_type not in {"image", "csv"}:
            raise ValueError(
                "Arg data_type must be an element of {image, csv}"
                f"You provided: {data_type}"
            )

        self._data_type = data_type


        if not isinstance(size, (int, tuple)):
            raise ("Argument Size takes values of type int or tuple")
            
        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError("Size Tuple should have two elements- height and weight")
        else:
            self._size = int(size)
        
    def _get_int_based_size_for_larger_dimension(self, bigger_size, small_size) -> int:
        
        bigger_size = bigger_size * self._size/small_size
        
        return int(bigger_size)
        
    def __call__(self, data: Dict):
        image = data["image"]
        key_points = data.get("facial_landmarks", None)
        if self._data_type == "image":

            current_height, current_weight = image.shape
            
            if isinstance(self._size, int):
                
                if current_height > current_weight:
                    
                    new_height = self._get_int_based_size_for_larger_dimension(
                        bigger_size=current_height, small_size=current_weight)
                    
                    new_weight = self._size
                    
                else:
                    new_weight = self._get_int_based_size_for_larger_dimension(
                        bigger_size=current_weight, small_size=current_height)
                    
                    new_height = self._size
                    
            else:
                new_height, new_weight = self._size
                
            return {
                "image": cv2.resize(image, (new_weight, new_height)), 
                "facial_landmarks": (
                    key_points * [new_weight/current_weight, new_height/current_height]
                    if key_points else key_points
                )
            }
            
        else:
            if isinstance(self._size, int):
                image = image.reshape(self._size, self._size, 1)
            else:
                image = image.reshape(self._size[0], self._size[1], 1)
            return {
                "image": image,
                "facial_landmarks": key_points
            }

class RandomCrop:
    
    def __init__(self, size: Union[int, Tuple[int, int]]):
        if not isinstance(size, (int, tuple)):
            raise ("Argument Size takes values of type int or tuple")
            
        self._size = int(size)
        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError("Size Tuple should have two elements- height and weight")
        else:
            self._size = (int(size), int(size))
            
    def __call__(self, sample):

        label_key: Optional[str] = None
        labels = None
        for key in sample:
            if key == "image":
                image = sample[key]
            else:
                label_key = key
                labels = sample[key]
        current_height, current_weight = image.shape
        
        new_h, new_w = self._size
        top = np.random.randint(0, current_height - new_h)
        left = np.random.randint(0, current_weight - new_w)

        sample = dict(image= image[top: top + new_h, left: left + new_w])
        if label_key:
            sample[label_key] = labels - [left, top] if labels else labels

        return sample
        
        
    
class ToTensor:
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample["image"]
        key_points = sample.get("facial_landmarks", None)
         
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        return {
            'image': torch.from_numpy(image),
            'facial_landmarks': torch.from_numpy(key_points) if key_points else None
        }