from typing import Tuple, Dict, Optional, Union
from abc import abstractmethod, ABCMeta
from src.cv.pytorch.models.configs import (
    ModelTrainingConfig, ModelDataConfig
)
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import torch
from matplotlib import pyplot as plt
import os
import logging
import json
import sys


root = logging.getLogger()
root.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s - %(asctime)s', datefmt='%d-%b-%y %H:%M:%S')
handler.setFormatter(formatter)
root.addHandler(handler)

class CNNTrainingPipeline(metaclass=ABCMeta):

    def __init__(
        self, 
        dataset: Dataset, 
        model_training_config: Dict, 
        model_data_config: Dict,
        load_model_from_path: Optional[Union[str, bool]] = None,
        model_initialization_params: Optional[Dict] = None,
    ):
        
        self.dataset = dataset
        self.model_training_config = ModelTrainingConfig(**model_training_config)
        self.model_data_config = ModelDataConfig(**model_data_config)
        self.__set_random_seed()
        
        if not load_model_from_path:
            required_non_essential_arguments = {
                "alpha_leaky_relu",
            }
            model_params_init_condition =  (
                model_initialization_params.get("cnn_batch_norm_flag", False) or 
                model_initialization_params.get("linear_batch_norm_flag", False)
            )
            if model_params_init_condition:
                required_non_essential_arguments.add(
                    "batch_norm_epsilon"
                )
                required_non_essential_arguments.add(
                    "batch_norm_momentum"
                )
            for key in required_non_essential_arguments:
                if key not in model_initialization_params:
                    model_initialization_params[key] = getattr(
                        self.model_training_config, key
                    )

            self.model = self._initialize_model(
                device=self.model_training_config.device,
                model_params=model_initialization_params,
            )
        else:
            if not isinstance(load_model_from_path, str):
                self.model = self.load_model_data(
                    device=self.model_training_config.device,
                    model_path=self.model_data_config.model_save_path
                )

            else:
                self.model = self.load_model_data(
                    device=load_model_from_path,
                    model_path=self.model_data_config.model_save_path
                )

        self._train_dataloader, self._validation_dataloader = self.__get_dataloader()
        self.optimizer, self.criterion = self.initialize_optimization_parameters(
            lr=self.model_training_config.learning_rate
        )
        self.train_loss = []
        self.validation_loss = []

        self._number_of_training_batches = int(
            self.model_data_config.train_size/self._train_dataloader.batch_size
        )

        if self._validation_dataloader:
            self._number_of_validation_batches = int(
                self.model_data_config.validation_size/self._validation_dataloader.batch_size
            )
        else:
            self._number_of_validation_batches = 0
        
        self._final_trained_model = None
        self._min_validation_loss: float = float(sys.maxsize)
        
    def __set_random_seed(self):

        np.random.seed(self.model_data_config.random_seed)
        torch.manual_seed(self.model_data_config.random_seed)
        torch.random.manual_seed(self.model_data_config.random_seed)
        if self.model_training_config.device == "cuda":
            torch.cuda.manual_seed(self.model_data_config.random_seed)
            torch.cuda.manual_seed_all(self.model_data_config.random_seed) 
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @abstractmethod
    def _initialize_model(self, device: str, model_params: Dict):
        pass

        raise NotImplementedError
        
    def __get_dataloader(self):

        indices = np.arange(self.model_data_config.dataset_size)
        if self.model_data_config.shuffle_dataset:
            np.random.shuffle(indices)
        train_indices = indices[:self.model_data_config.train_size]
        validation_dataloader = None
        if self.model_data_config.shuffle_dataset:
            train_sampler = SubsetRandomSampler(train_indices)
            train_dataloader = DataLoader(
                self.dataset, 
                batch_size=self.model_training_config.batch_size, 
                num_workers=mp.cpu_count() - 2, 
                sampler=train_sampler
            )
            if self.model_data_config.validation_size:
                validation_indices = indices[self.model_data_config.train_size:]
                validation_sampler = SubsetRandomSampler(validation_indices)
                validation_dataloader = DataLoader(
                    self.dataset, 
                    batch_size=self.model_training_config.batch_size, 
                    num_workers=mp.cpu_count() - 2, 
                    sampler=validation_sampler
                )

        else:
            train_data = [self.dataset[i] for i in validation_indices]
            train_dataloader = DataLoader(
                train_data,
                shuffle=False,
                batch_size=self.model_training_config.batch_size, 
                num_workers=mp.cpu_count() - 2, 
            )
            if self.model_data_config.validation_size:
                validation_data = [self.dataset[i] for i in validation_indices]
                validation_dataloader = DataLoader(
                    validation_data,
                    shuffle=False,
                    batch_size=self.model_training_config.batch_size, 
                    num_workers=mp.cpu_count() - 2, 
                )
        
        return train_dataloader, validation_dataloader

    @abstractmethod
    def _fit_model(self):
        """
        Function to fit the model to training data
        
        """
        
        raise NotImplementedError

    @abstractmethod
    def _validate_model(self):
        """
        Function to evaluate model on vlidation data is available
        """
        raise NotImplementedError


    def train(self):
        """
        """
        self.model.to(self.model_training_config.device)
        for epoch in tqdm(range(self.model_training_config.epochs), total=self.model_training_config.epochs):
            epoch_train_loss = self._fit_model()
            logging.info(f"Train Loss for epoch {epoch}: {epoch_train_loss:.4f}")
            self.train_loss.append(epoch_train_loss)
            if self._number_of_validation_batches:
                epoch_validation_loss = self._validate_model()
                logging.info(f"Validation Loss for epoch {epoch}: {epoch_validation_loss:.4f}")
                self.validation_loss.append(epoch_validation_loss)


    def generate_train_validation_loss_curves(self, save_figure=False):
        plt.figure(figsize=(10, 7))
        plt.plot(self.train_loss, color='blue', label='train loss through epochs')
        plt.plot(self.validation_loss, color='red', label='validataion loss through epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        if save_figure:
            plt.savefig(f"{self.model_data_config.loss_curve_path}")
        plt.show()
        
    def save_model_data(self, model_path):
        torch.save(
            {
                'epoch': self.model_training_config.epochs,
                'model_state_dict': self.best_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.criterion,
            }, 
            os.path.join(
                self.model_data_config.model_save_path, model_path)
        )
        os.makedirs(ModelDataConfig.model_save_path, exist_ok = True)

    @staticmethod
    def load_model_data(model_path, device):
        
        with open(model_path, "r") as f:
            if device == "cuda":
                model = torch.load(f, map_location=lambda storage, loc: storage.cuda(0))
            else:
                model = torch.load(f, map_location=lambda storage, loc: storage)
        return model


    @abstractmethod
    def initialize_optimization_parameters(self, lr) -> Tuple:
        """
        
        """
        
        raise NotImplementedError

    @abstractmethod
    def get_predictions(self, test_dataloader):
        ""
        raise NotImplemented

    @property
    def best_model(self):
        if self._final_trained_model:
            return self._final_trained_model

        else:
            raise ValueError("Pleass train the model to get the best estimator")