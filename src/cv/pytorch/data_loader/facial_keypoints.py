import os
import torch
import numpy as np
from PIL import Image
from skimage import io
import cv2
from matplotlib import pyplot as plt
from torchvision import transforms
from src.cv.pytorch.data_loader.dataloader import CustomDatasetLoader
from typing import Dict, Union, Tuple, Optional


class Grayscale:
    
    def __init__(self, data_type: str):

        if data_type not in {"image", "csv"}:
            raise ValueError(
                "Arg data_type must be an element of {image, csv}"
                f"You provided: {data_type}"
            )

        self._data_type = data_type

    def __call__(self, data: Dict):
        
        if self._data_type == "image":
            image = data["image"]
            key_points = data["facial_landmarks"]
            
            # rescaling [0 to 1] from [0, 255]
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)/255
            return {'image': grayscale_image, "facial_landmarks": key_points}
        
        else:
            return {"image": data["image"]/255, "facial_landmarks": key_points}
    

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

        if self._data_type == "image":
        
            image = data["image"]
            key_points = data["facial_landmarks"]
            
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
                "facial_landmarks": key_points * [new_weight/current_weight, new_height/current_height]
            }
            
        else:

            if isinstance(self._size, int):
                return {
                    "image": image.resize(self._size, self._size, 1),
                    "facial_landmarks": key_points
                }
            else:
                return {
                    "image": image.resize(self._size[0], self._size[1], 1),
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
        image = sample["image"]
        key_points = sample["facial_landmarks"]
        current_height, current_weight = image.shape
        
        new_h, new_w = self._size
        top = np.random.randint(0, current_height - new_h)
        left = np.random.randint(0, current_weight - new_w)

        return {
            'image': image[top: top + new_h, left: left + new_w], 
            'facial_landmarks': key_points - [left, top]
        }

    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample["image"]
        key_points = sample["facial_landmarks"]
         
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
            'facial_landmarks': torch.from_numpy(key_points)
        }

class FacialKeypointDataLoader(CustomDatasetLoader):

    def __init__(
        self, 
        data_type: str,
        dataset_name: str, 
        img_dir: Optional[str] = None, 
        image_column: Optional[str] = None,
        annotation_file: Optional[str] = None, 
        train_data: Optional[str] = None,
        test_data: Optional[str] = None
    ) -> None:
        
        if self._data_type == "image":
            super().__init__(
                data_type = data_type,
                dataset_name=dataset_name, 
                img_dir=img_dir, 
                annotation_file=annotation_file, 
                composed_transforms=transforms.Compose([
                    Grayscale(data_type=data_type),
                    Resize(data_type=data_type, size=256),
                    RandomCrop(224),
                    ToTensor(),
                ])
            )
        
        else:

            super().__init__(
                dataset_name=dataset_name,
                train_data=train_data,
                test_data=test_data,
                composed_transforms=transforms.Compose([
                    Resize(data_type=data_type, size=96),
                    Grayscale()
                ])
            )

        if data_type != "image":

            if not image_column:
                if "Image" not in self.dataset.image_labels.columns:
                    raise ValueError(
                        "No way to identify column containing image data"
                        "Please provide value in image_column argument "
                        "or put image data is 'Image' column"
                    )
                image_column = "Image"

            if not isinstance(self.dataset.image_labels.loc[0, image_column], np.ndarray):
                self.dataset.image_labels.Image.apply(
                    lambda x: np.array(x.split(" "), dtype="float"))
            self.dataset.image_data = self.dataset.image_labels.pop(image_column)
            self._image_labels = self.dataset.image_labels

    def _get_image_data(self, idx):

        # getting image name from keypoint file
        img_name = os.path.join(self.dataset.img_directory, self.dataset.image_labels.iloc[idx, 0])
        # reading input file in
        image = io.imread(img_name)
        
        # extracting keypoints for the image
        image_label = np.array(
            self.dataset.image_labels.iloc[idx, 1:]).astype("float").reshape(-1, 2)
        sample = {'image': image, 'facial_landmarks': image_label}

        sample = self.transform(sample)

        return sample

    def __getitem__(self, idx):
        """ 
        Function copied from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        
        """
        # get idx to be returned as a list
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self._data_type == "image":
            return self._get_image_data(self, idx)

        else:
            image = self.dataset.image_data[idx]
            keypoints = self.dataset.image_labels.iloc[idx, :]

            sample = {"image": image, "facial_landmarks": keypoints}
            sample = self.transform(sample)
            return sample


    def show_key_points_on_images(
        self, 
        idx, 
    ):
        """
        Function plots image and associated keypoint on the image for the input idx
        """

        keypoints = np.asarray(
            self.dataset.image_labels.iloc[idx, 1:]
        ).astype('float').reshape(-1, 2)

        if self._data_type == "image":
            loaded_image = io.imread(
                os.path.join(
                    self.dataset.img_directory, 
                    self._image_labels.iloc[idx, 0]
                )
            )

        else:
            loaded_image = self.dataset.image_data[idx]

        plt.figure()
        plt.imshow(loaded_image)
        plt.scatter(keypoints[:, 0], keypoints[:, 1], s=10, c='b', marker="*")

        plt.pause(0.002)

        plt.show()
    