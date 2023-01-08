import os
import torch
import numpy as np
from PIL import Image
from skimage import io
import cv2
from matplotlib import pyplot as plt
from torchvision import transforms
from ..data_loader.dataloader import CustomDatasetLoader
from typing import Dict, Union, Tuple, Optional


class Grayscale:
    
    def __call__(self, data:Dict):
        
        image = data["image"]
        key_points = data["facial_landmarks"]
        
        # rescaling [0 to 1] from [0, 255]
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)/255
        return {'image': grayscale_image, "facial_landmarks": key_points}
    

class Resize:
    
    def __init__(self, size: Union[int, Tuple[int, int]]):
        
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
        dataset_name: str, 
        img_dir: str, 
        annotation_file: str = None, 
    ) -> None:

        
        super().__init__(
            dataset_name=dataset_name, 
            img_dir=img_dir, 
            annotation_file=annotation_file, 
            composed_transforms=transforms.Compose([
                Grayscale(),
                Resize(256),
                RandomCrop(224),
                ToTensor(),
            ])
        )


    def __getitem__(self, idx):
        """ 
        Function copied from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        
        """
        
        # get idx to be returned as a list
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # getting image name from keypoint file
        img_name = os.path.join(self.dataset.img_directory, self.dataset.image_labels.iloc[idx, 0])
        # reading input file in
        image = io.imread(img_name)
        
        # extracting keypoints for the image
        image_label = np.array(
            self.dataset.image_labels.iloc[idx, 1:]).astype("float").reshape(-1, 2)
        sample = {'image': image, 'facial_landmarks': image_label}

        # it any transforms are present - apply transforms to the extracted image
        if self.transform:
            sample = self.transform(sample)
#             sample["image"] = Image.fromarray(sample["image"])

        return sample

    def show_key_points_on_images(
        self, 
        idx, 
        epoch:Optional[int] = None, 
        predicted_keypoints: Optional[np.array] = None):

        """
        Function plots image and associated keypoint on the image for the input idx
        """

        keypoints = np.asarray(
            self._image_labels.iloc[idx, 1:]
        ).astype('float').reshape(-1, 2)

        # if predicted_keypoints:
        #     if not epoch:
        #         raise ValueError("Epoch Argument needed to show predicted keypoints")
        #     predicted_keypoint_to_use = predicted_keypoints.detach().cpu().numpy()
        #     predicted_keypoint_to_use = np.asarray(
        #         predicted_keypoint_to_use.iloc[idx, 1:]
        #     ).reshape(-1, 2)

        loaded_image = io.imread(
            os.path.join(
                self.dataset.img_directory, 
                self._image_labels.iloc[idx, 0]
            )
        )

        plt.figure()
        plt.imshow(loaded_image)
        plt.scatter(keypoints[:, 0], keypoints[:, 1], s=10, c='b', marker="*")
        # if predicted_keypoints:
        #     plt.scatter(
        #         predicted_keypoint_to_use[:, 0], 
        #         predicted_keypoint_to_use[:, 1],
        #         s = 10.
        #         c = "r"
        #         marker"*"
        #     )
        #     plt.title(f"Original and predicted keypoints for {idx} and epoch {epoch}")

        plt.pause(0.002)

        plt.show()
    