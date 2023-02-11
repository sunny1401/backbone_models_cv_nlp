import pydicom
import numpy as np
import torch
import os
from src.cv.pytorch.datasets.base_dataset import CustomDataset


class BreastCancerDataset(CustomDataset):
    
    def __init__(
        self, 
        annotation_file: str, 
        img_dir: str, 
        composed_transforms=None,
        is_test: bool = False, 
    ):
        
        super().__init__(
            dataset_name = "rsna_breast_cancer_dataset",
            data_type="image",
            is_test=is_test,
            annotation_file=annotation_file,
            img_dir=img_dir,
            composed_transforms=composed_transforms
            
        )
        
    def __parse_image(self, img: pydicom.dataset.FileDataset):
        
        image_array = img.pixel_array
        if img.PhotometricInterpretation == "MONOCHROME1":
            # normalize across pixel array to make difference prominent
            image_array = np.max(image_array) - image_array
        
        return image_array
    
    def __len__(self):
        
        # while the images themselves are multiple -> 
        # either a patient will have cancer or won't.
        # information used for creating train and validation dataloader
        return self.dataset.image_labels.patient_id.nunique()
        
        
    def __getitem__(self, idx):
        
        # get idx to be returned as a list
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        required_columns = [
            "cancer", "image_id", "patient_id", "laterality"
        ]
        
        
        annotation_data = self.dataset.image_labels.loc[idx, required_columns]
        patient_id = str(int(annotation_data["patient_id"]))
        image_id = str(int(annotation_data["image_id"]))
        image_path = f"{os.path.join(self.dataset.img_directory, patient_id, image_id)}.dcm"
        image = self.__parse_image(pydicom.read_file(image_path))
            
        sample = dict(
            image=image, 
            label=annotation_data["cancer"], 
            lateral_view=annotation_data["laterality"]
        )

        if self.transform:
            sample = self.transform(sample)
            sample = dict(
                image=sample["image"][np.newaxis],
                label=sample["label"]
            )


        if sample["image"].shape[0] == 1:
            # todo - do this correctly
            sample["image"] = np.reshape(sample["image"], (sample["image"].shape[1], sample["image"].shape[2], sample["image"].shape[0]))
            
        return sample
