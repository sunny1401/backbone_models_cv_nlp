from typing import Dict, Union, Tuple
import cv2

class Resize:

    def __init__(self, size: Union[int, Tuple[int, int]], do_center_crop: bool = True) -> None:
        # Store size and center_crop values in class object
        if do_center_crop == True:
            if isinstance(size, Tuple) and size[0] != size[1]:
                raise ValueError("The values passed for arguments size and centering is confusing."
                                 "For centering == True, "
                                 "width information is taken from the passed input image")
            self._size = size if isinstance(size, int) else size[0]
        else:
            self._size = size if isinstance(size, Tuple) else (size, size)
        self._centering = do_center_crop

    def __get_center_coordinates(self, image, lateral_view):
        _, width = image.shape[:2]

        # Calculate the center of the image
        center = int(width//2)
        if lateral_view == "L":
            # Crop the left half of the image
            image = image[:, :center]
        else:
            # Crop the right half of the image
            image = image[:, center:]
        return image

    def __call__(self, sample: Dict) -> Dict:
        image = sample["image"]

        height, _ = image.shape

        if self._centering:

            laterality = sample["lateral_view"]
            center_cropped_image = self.__get_center_coordinates(
                image=image, lateral_view=laterality
            )
            
            resized_img = cv2.resize(
                center_cropped_image, (self._size, height),
                            interpolation=cv2.INTER_CUBIC)

        else:
            resized_img = cv2.resize(
                image, self._size
            )

        return dict(
            image= resized_img,
            label= sample["label"],
            lateral_view = sample["lateral_view"]
        )


class Denioser:

    def __init__(
        self, 
        kernel_size: Union[int, Tuple[int. int]], 
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
            image = cv2.GaussianBlur(image, self.kernel, 0)

        sample["image"] = image
        return sample

        
class HistogramEqualization:

    def __init__(
        self, 
        clip_limit: float = 2.0, 
        tile_grid_size: Tuple[int, int]=(8,8)
    ):

        if not isinstance(tile_grid_size, Tuple):
            raise ValueError(
                "tile_grid_size should be of type Tuple"
            )

        self._tile_grid_size = tile_grid_size
        self._clip_limit = clip_limit
        self._histogram_equalizer = cv2.createCLAHE(
            clipLimit=self._clip_limit, 
            tileGridSize=self._tile_grid_size
        )

    def __call__(self, sample: Dict) -> Dict:
        
        sample["image"] = self._histogram_equalizer.apply(
            sample["image"]
        )
        return sample
