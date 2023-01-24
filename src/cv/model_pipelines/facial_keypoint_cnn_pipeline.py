from src.cv.model_pipelines.dl_base_pipeline import CNNTrainingPipeline
from src.cv.pytorch.models.use_cases.facial_keypoint_detection.facial_keypoint_vanilla_cnn import FacialKeypointVCNN
from typing import Dict, List, Optional, Union
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import sys

class FacialCNNTrainingPipeline(CNNTrainingPipeline):


    def __init__(self, 
        dataset: Dataset, 
        model_training_config: Dict, 
        model_data_config: Dict, 
        model_initialization_params: Optional[Dict] = None,
        load_model_from_path: Optional[Union[bool, str]] = False,
    ):
          super().__init__(
            dataset=dataset, 
            model_training_config=model_training_config, 
            model_data_config=model_data_config, 
            model_initialization_params=model_initialization_params,
            load_model_from_path=load_model_from_path
        )

    def initialize_optimization_parameters(self, lr=0.0005) -> Dict:
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.model.parameters(), lr=lr
        )
        
        return optimizer, criterion

    def _initialize_model(self, device, model_params):
        model = FacialKeypointVCNN(**model_params).to(device)
        return model

    def _fit_model(self):

        self.model.train()
        batch_training_loss: float = 0
        
        for idx, data in enumerate(self._train_dataloader):
            image = data["image"]
            keypoints = data["facial_landmarks"]
            # flatten pts
            keypoints = keypoints.view(keypoints.size(0), -1)
            # convert variables to floats for regression loss
            keypoints = keypoints.type(torch.FloatTensor)
            image = image.type(torch.FloatTensor)
            # inp is shape (N, C, H, W)
            image = image.reshape(image.shape[0], image.shape[-1], image.shape[1], image.shape[2])
            image = image.to(self.model_training_config.device)
            keypoints = keypoints.to(self.model_training_config.device)
            self.optimizer.zero_grad()
            outputs = self.model(image)
            loss = self.criterion(
                outputs, 
                keypoints
            )
            batch_training_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        
        train_loss = batch_training_loss/ idx +1
        return train_loss

    def _validate_model(self):
        batch_validation_loss: float = 0
        self.model.eval()
        with torch.no_grad():

            for idx, data in enumerate(self._validation_dataloader):
                image = data["image"]
                keypoints = data["facial_landmarks"]
                # flatten pts
                keypoints = keypoints.view(keypoints.size(0), -1)
                # convert variables to floats for regression loss
                keypoints = keypoints.type(torch.FloatTensor)
                image = image.type(torch.FloatTensor)
                 # inp is shape (N, C, H, W)
                image = image.reshape(image.shape[0], image.shape[-1], image.shape[1], image.shape[2])
                image = image.to(self.model_training_config.device)
                keypoints = keypoints.to(self.model_training_config.device)
                outputs = self.model(image)
                loss = self.criterion(
                    outputs, 
                    keypoints
                )
                batch_validation_loss += loss.item()

        validation_loss = batch_validation_loss/idx + 1
        if self._min_validation_loss > validation_loss:
            self._min_validation_loss = validation_loss
            self._final_trained_model = self.model
        return validation_loss

    def get_predictions(self, test_dataloader, device) -> List:
        
        model = self.best_model
        if model.get_device() != device:
            model.to(device)
        model.eval()

        output_keyp = []
        for i in range(len(test_dataloader)):
            data = test_dataloader.dataset[i]
            image = data["image"].to(device)
            image = image.type(torch.FloatTensor)
                # inp is shape (N, C, H, W)
            image = image.reshape(image.shape[0], image.shape[-1], image.shape[1], image.shape[2])
            output_keyp.append(model(image))

        return output_keyp