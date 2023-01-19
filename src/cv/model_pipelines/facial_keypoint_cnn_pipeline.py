from src.cv.model_pipelines.dl_base_pipeline import CVTrainingPipeline
from src.cv.pytorch.models.use_cases.facial_keypoint_detection.facial_keypoint_vanilla_cnn import FacialKeypointVCNN
from typing import Dict
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from tqdm import tqdm


class FacialCNNTrainingPipeline(CVTrainingPipeline):


    def __init__(self, dataset: Dataset, model_training_config: Dict, model_data_config: Dict, model_initialization_params: Dict = None):
          super().__init__(dataset, model_training_config, model_data_config, model_initialization_params)

    def initialize_optimization_parameters(self, lr=0.0005) -> Dict:
        
        optimization_functions = dict()
        optimization_functions["criterion"] = nn.MSELoss()
        optimization_functions["optimizer"] = optim.Adam(
            self.parameters(), lr=lr
        )
        
        return optimization_functions

    def __initialize_model(self, model_params):
        model = self.__initialize_model(**model_params)
        return model

    def __validate_model(self):
        batch_validation_loss: float = 0
        self.model.eval()
        counter: int = 0
        with torch.no_grad():

            for idx, data in tqdm(
                enumerate(self._validation_dataloader),
                total=self._number_of_validation_batches
            ):
                counter += 1
                image = data["image"].to(self.model_training_config.device)
                keypoints = data["facial_landmarks"].to(self.model_training_config.device)
                # flatten pts
                keypoints = keypoints.view(keypoints.size(0), -1)
                # convert variables to floats for regression loss
                keypoints = keypoints.type(torch.FloatTensor)
                images = images.type(torch.FloatTensor)
                outputs = self.model(image)
                loss = self.criterion(outputs, keypoints)
                batch_validation_loss += loss.item()
            print(idx, counter)

        validation_loss = batch_validation_loss/counter
        return validation_loss

    def __fit_model(self):

        self.model.train()
        batch_training_loss: float = 0
        
        for idx, data in tqdm(
            enumerate(self._train_dataloader), 
            total= self._number_of_training_batches
        ):
            image = data["image"].to(self.model_training_config.device)
            keypoints = data["facial_landmarks"].to(self.model_training_config.device)
            # flatten pts
            keypoints = keypoints.view(keypoints.size(0), -1)
            # convert variables to floats for regression loss
            keypoints = keypoints.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)
            self.optimizer.zero_grad()
            outputs = self.model(image)
            loss = self.criterion(outputs, keypoints)
            batch_training_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        
        train_loss = batch_training_loss/ idx +1
        return train_loss
