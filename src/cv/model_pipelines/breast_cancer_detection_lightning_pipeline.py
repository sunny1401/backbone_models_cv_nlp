from src.cv.model_pipelines.dl_base_pytorch_lightning_pipeline import CNNTrainingPLPipeline
from src.cv.pytorch.models.resnet import VanillaResnet
from typing import Callable, Dict, Optional
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import numpy as np
import multiprocessing as mp


class BreastCancerLightningPipeline(CNNTrainingPLPipeline):

    def __init__(
        self, model: Callable, 
        dataset: Dataset, 
        model_training_config: Dict, 
        model_data_config: Dict, 
        model_initialization_params: Dict, 
        train_on_full_dataset: bool = False,
        loss_function_weights: Optional[np.array] = None, 
        load_model_from_path: Optional[str] = None

    ) -> None:
        super().__init__(
            model, 
            dataset, 
            model_training_config, 
            model_data_config, 
            model_initialization_params, 
            loss_function_weights, 
            load_model_from_path
        )
        self._train_on_full_dataset = train_on_full_dataset
        self._train_indices, self._val_indices = self._generate_train_validation_indices()

    def _get_model_from_params(self, model: Callable, hyper_params: Dict):

        other_required_keys = dict(
            dropout_threshold=0.09,
            num_linear_layers=1,
            output_channels=None,
            add_dropout_to_linear_layers=True
        
        )
        wraping_layers_map = {}
        for key, value in other_required_keys.items():
            wraping_layers_map[key] = hyper_params.pop(key, value)

        model = VanillaResnet(**hyper_params).to(self.model_training_config.device)
        model.wrap_up_network(**wraping_layers_map)
        return model

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(), **self._optimizer_params
        )
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        
        return [optimizer], [scheduler]

    def _generate_train_validation_indices(self):
        columns_required = ["image_id", "patient_id", "laterality", "cancer"]
        df = self.dataset.dataset.image_labels[columns_required]
        if not self._train_on_full_dataset and self.model_data_config.train_data_pct < 1:
            test_pct = 1 - self.model_data_config.train_data_pct

            df["patient_id"] = df.apply(
                lambda row: f"{row['patient_id']}_{row['laterality']}", axis=1).drop(columns=["laterality"])

            num_cancer_val_ids = round(df[df.cancer == 1]["patient_id"].nunique() * test_pct)
            num_ncancer_val_ids = round(df[df.cancer == 0]["patient_id"].nunique() * test_pct)
            cancer_val_ids = list(np.random.choice(df[df.cancer == 1]["patient_id"].unique(), num_cancer_val_ids))
            num_ncancer_val_ids = list(np.random.choice(df[df.cancer == 0]["patient_id"].unique(), num_ncancer_val_ids))

            val_ids = cancer_val_ids + num_ncancer_val_ids
            train_ids = [patient_id for patient_id in df.patient_id.unique() if patient_id not in val_ids]
            val_indices = df[df.patient_id.isin(val_ids)].index
            train_indices = df[df.patient_id.isin(train_ids)].index

        else:
            train_indices = df.index.tolist()
            val_indices = None

        return train_indices, val_indices

    def train_dataloader(self):
        if self.model_data_config.shuffle_dataset:
            train_sampler = SubsetRandomSampler(self._train_indices)
            train_dataloader = DataLoader(
                self.dataset, 
                batch_size=self.model_training_config.batch_size, 
                num_workers=mp.cpu_count() - 2, 
                sampler=train_sampler
            )

        else:
            train_data = [self.dataset[i] for i in self._train_indices]
            train_dataloader = DataLoader(
                train_data,
                shuffle=False,
                batch_size=self.model_training_config.batch_size, 
                num_workers=mp.cpu_count() - 2, 
            )
        return train_dataloader
        
        
    def val_dataloader(self):
        if self._val_indices is not None:
            validation_sampler = SubsetRandomSampler(self._val_indices)
            validation_dataloader = DataLoader(
                self.dataset, 
                batch_size=self.model_training_config.batch_size, 
                num_workers=mp.cpu_count() - 2, 
                sampler=validation_sampler
            )
        else:
            validation_data = [self.dataset[i] for i in self._val_indices]
            validation_dataloader = DataLoader(
                validation_data,
                shuffle=False,
                batch_size=self.model_training_config.batch_size, 
                num_workers=mp.cpu_count() - 2, 
            )
        return validation_dataloader

    def test_dataloader(self):
        dataloader = []
        index = 0
        dataset_finished = False
        current_end = self.model_training_config.batch_size
        if not len(self._test_dataset):
            return None
        while(current_end <= len(self._test_dataset)):
            dataloader.append(torch.from_numpy(np.array(
                [self._test_dataset[j]["image"] for j in range(index, current_end)]
            )))
            if dataset_finished:
                break
            index = current_end
            current_end += self.model_training_config.batch_size
            if current_end > len(self._test_dataset):
                current_end = len(self._test_dataset)
                dataset_finished = True

        return dataloader

    def forward(self, x):
        return self.model.forward(x)