from typing import Tuple, Dict
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


torch.backends.cudnn.deterministic = True
class CVTrainingPipeline(ABCMeta):

    def __init__(
        self, 
        dataset: Dataset, 
        model_training_config: Dict, 
        model_data_config: Dict,
        model_initialization_params: Dict = None,
    ):
        
        self.model = self.__initialize_model(
            model_initialization_params
        )
        self.__set_random_seed()
        self.dataset = dataset
        self.model_training_config = ModelTrainingConfig(**model_training_config)
        self.model_data_config = ModelDataConfig(**model_data_config)

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
    def __initialize_model(self, model_params):
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

    # number_of_batches = int(len(self.dataset))
    @abstractmethod
    def __fit_model(self):
        """
        
        """
        
        raise NotImplementedError

    @abstractmethod
    def __validate_model(self):
        """
        """
        raise NotImplementedError


    def train(self):
        """
        """
        
        for epoch in self.model_training_config.epochs:
            epoch_train_loss = self.__fit_model()
            print(f"Train Loss for epoch {epoch}: {epoch_train_loss:.4f}")
            self.train_loss.append(epoch_train_loss)
            if self._number_of_validation_batches:
                epoch_validation_loss = self.__validate_model()
                print(f"Train Loss for epoch {epoch}: {epoch_validation_loss:.4f}")
                self.validation_loss.append(epoch_validation_loss)


    def generate_train_validation_loss_curves(self, save_figure=True):
        plt.figure(figsize=(10, 7))
        plt.plot(self.train_loss, color='blue', label='train loss through epochs')
        plt.plot(self.validation_loss, color='red', label='validataion loss through epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"{self.model_data_config.loss_curve_path}")
        plt.show()
        
    def save_model_data(self, model_path):
        torch.save(
            {
                'epoch': self.model_training_config.epochs,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.criterion,
            }, 
            os.path.join(
                ModelDataConfig.model_save_path, model_path)
        )
        os.makedirs(ModelDataConfig.model_save_path, exist_ok = True)


    @abstractmethod
    def initialize_optimization_parameters(self, lr) -> Tuple:
        """
        
        """
        
        raise NotImplementedError
