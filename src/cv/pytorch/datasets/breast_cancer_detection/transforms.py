from typing import Dict, Union, Tuple
import cv2
import numpy as np

class Resize:

    def __init__(
        self, size: Union[int, Tuple[int, int]], 
        zoom_factor=1.5, 
        lateral_crop: bool = True
    ) -> None:
        # Store size and center_crop values in class object
        if lateral_crop == True:
            if isinstance(size, Tuple) and size[0] != size[1]:
                raise ValueError("The values passed for arguments size and centering is confusing."
                                 "For centering == True, "
                                 "width information is taken from the passed input image")
            self._size = size if isinstance(size, int) else size[0]
        else:
            self._size = size if isinstance(size, Tuple) else (size, size)
        self._lateral_crop = lateral_crop
        self._zoom_factor = zoom_factor

    def __get_center_coordinates(self, image, lateral_view):
        _, width = image.shape[:2]

        # Calculate the center of the image
        if lateral_view == "L":
            # Crop the left half of the image
            new_width = int(width*0.75)
            image = image[:, :new_width]
        else:
            new_width = int(width*0.25)
            # Crop the right half of the image
            image = image[:, new_width:]
        return image

    def __call__(self, sample: Dict) -> Dict:
        image = sample["image"]

        height, _ = image.shape

        if self._lateral_crop:

            laterality = sample["lateral_view"]
            center_cropped_image = self.__get_center_coordinates(
                image=image, lateral_view=laterality
            )

            resized_img = cv2.resize(
                center_cropped_image, 
                (self._size, height),
                interpolation=cv2.INTER_CUBIC, 
                fx=self._zoom_factor, fy=self._zoom_factor
            )

        else:
            resized_img = cv2.resize(
                image, self._size
            )

        return dict(
            image= resized_img,
            label= sample["label"],
            lateral_view = sample["lateral_view"]
        )


class Denoiser:

    def __init__(
        self, 
        kernel_size: Union[int, Tuple[int, int]], 
        blur_type: str = "gaussian"
    ) -> None:
        
        if blur_type not in {"possion", "gaussian"}:
            raise ValueError(
                "Only Gaussian and Poisson Blurs supported for now"
            )

        if blur_type == "poisson":
            if isinstance(kernel_size, Tuple) and kernel_size[0] != kernel_size[1]:

                raise ValueError(
                    "The values passed for arguments blur_type and kernel_size are confusing."
                    "For blur_type == poisson "
                    "kernel should be either int or widht and height be of same size"
                )

            self._kernel = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        else:
            self._kernel = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size

        self._blur_type = blur_type

    def __call__(self, sample: Dict) -> Dict:
        

        image = sample["image"]
        if self._blur_type == "poisson":
            image = cv2.MediumBlur(image, self._kernel)

        else:
            image = cv2.GaussianBlur(image, self._kernel, 0)

        sample["image"] = image
        return sample

        
class HistogramEqualization:

    def __init__(
        self, 
        number_bins=256
    ):

        self._number_bins = number_bins
    # from http://www.janeriksolem.net/histogram-equalization-with-python-and.html

    def __call__(self, sample: Dict) -> Dict:
        
        image = sample["image"]
        # get image histogram
        image_histogram, bins = np.histogram(image.flatten(), self._umber_bins, density=True)
        cdf = image_histogram.cumsum() # cumulative distribution function
        cdf = (self._number_bins-1) * cdf / cdf[-1] # normalize

        # use linear interpolation of cdf to find new pixel values
        image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

        sample["image"] = image_equalized
