import pytorch_lightning as pl
from torch import nn
from typing import Callable, Dict, Optional
from src.cv.pytorch.models.configs import (
    ModelTrainingConfig, ModelDataConfig
)
from torch.utils.data import Dataset
from abc import abstractmethod
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import torch

from matplotlib import pyplot as plt
import os
import logging

import sys


root = logging.getLogger()
root.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s - %(asctime)s', datefmt='%d-%b-%y %H:%M:%S')
handler.setFormatter(formatter)
root.addHandler(handler)


class CNNTrainingPLPipeline(pl.LightningModule):


    def __init__(
        self, 
        model: Callable,
        dataset: Dataset, 
        model_training_config: Dict, 
        model_data_config: Dict,
        model_initialization_params: Dict,
        optimizer_params: Dict,
        loss_function_weights: Optional[np.array] = None, 
        load_model_from_path: Optional[str] = None,
        test_dataset: Optional[Dataset] = None
    ) -> None:
        super(CNNTrainingPLPipeline, self).__init__()

        self.dataset = dataset
        self.model_training_config = ModelTrainingConfig(**model_training_config)
        self.model_data_config = ModelDataConfig(**model_data_config)
        self.model_training_config = ModelTrainingConfig(**model_training_config)
        self._optimizer_params = optimizer_params
        self.model_data_config = ModelDataConfig(**model_data_config)
        self.__set_random_seed()
        self.model = self._get_model_from_params(model = model, hparams = model_initialization_params)
        self.loss_module = self._get_criterion(weights=loss_function_weights)
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        self._test_dataset = test_dataset


    def __set_random_seed(self):

        np.random.seed(self.model_data_config.random_seed)
        torch.manual_seed(self.model_data_config.random_seed)
        torch.random.manual_seed(self.model_data_config.random_seed)
        os.environ['PYTHONHASHSEED'] = str(self.model_data_config.random_seed)
        if self.model_training_config.device == "cuda":
            torch.cuda.manual_seed(self.model_data_config.random_seed)
            torch.cuda.manual_seed_all(self.model_data_config.random_seed) 
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        pl.seed_everything(self.model_data_config.random_seed)

    @abstractmethod
    def configure_optimizers(self, learning_rate: float):

        raise NotImplementedError

    @abstractmethod
    def _get_criterion(self, weights: Optional[np.array] = None):
        criterion = nn.BCEWithLogitsLoss(weights)
        return criterion

    @abstractmethod
    def _get_model_from_params(self, model: Callable, hyper_params: Dict):
        raise NotImplementedError

    @abstractmethod
    def forward(self):
        raise NotImplementedError

    def _init_params(self):
        # Based on our discussion in Tutorial 4, we should initialize the
        # convolutions according to the activation function
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity=self.hparams.act_fn_name)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @abstractmethod
    def train_dataloader(self):
        raise NotImplementedError

    @abstractmethod
    def val_dataloader(self):
        raise NotImplementedError

    def prepare_data(self):
        pass

    def test_dataloader(self):
        pass

    def training_step(self, train_batch, batch_idx):
      x, y = train_batch
      logits = self.forward(x)
      loss = self.loss_module(logits, y)

      logs = {'train_loss': loss}
      return {'loss': loss, 'log': logs}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.loss_module(logits, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        # called at the end of the validation epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}] 
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
