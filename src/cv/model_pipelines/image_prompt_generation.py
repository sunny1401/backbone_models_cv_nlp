import os
from typing import Dict, Tuple, Optional
import multiprocessing as mp

import numpy as np
from torch.utils.data import (
    random_split,
    Dataset,
    DataLoader
)
import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.cuda import free_gpu_cache
from src.cv.model_pipelines.dl_base_pipeline import CNNTrainingPipeline

from src.cv.pytorch.models.image_prompt_generation.vision_text_enc_dec import (
    Gpt2Decoder,
    VisionTextModel,
    VisionTransformer
)



class ImagePromptGeneration(CNNTrainingPipeline):

    def __init__(self, 
        dataset: Dataset, 
        model_training_config: Dict, 
        model_data_config: Dict, 
        model_initialization_params: Dict,
        load_model_from_path: Optional[str] = None
    ):
        self._model_path = load_model_from_path


        super().__init__(
            dataset=dataset, 
            model_training_config=model_training_config, 
            model_data_config=model_data_config, 
            model_initialization_params=model_initialization_params,
            load_model_from_path=load_model_from_path
        )

    def _initialize_model(self, device: str, model_params: Dict):

        text_model = Gpt2Decoder(
            existing_model_path=model_params["text_model_path"]
        )
        vision_model = VisionTransformer(
            image_size=model_params["image_size"],
            existing_model_path=model_params["vision_model_path"]
        )
        model = VisionTextModel(
            text_model=text_model,
            image_model=vision_model
        )

        model = nn.DataParallel(model).to(device)

        return model
    
    def _collate_data_func(self, batch):

        image_features, input_ids, attention_mask = zip(*batch)
        # Flatten the firt two dimensions from image_features

        image_features = torch.stack(image_features).view(-1, *image_features[0].shape[1:])
    
        input_ids = torch.stack([torch.tensor(x) for x in input_ids]).view(-1, input_ids[0].shape[1])
        attention_mask = torch.stack(attention_mask).view(-1, attention_mask[0].shape[1])
        return image_features, input_ids, attention_mask
    
    def get_dataloader(self, train_on_full_dataset: bool = False):

        generator = torch.Generator()
        generator.manual_seed(self.model_data_config.random_seed)

        train_data, test_data = random_split(
            dataset = self.dataset,
            lengths = [
                self.model_data_config.train_size, 
                self.model_data_config.validation_size
            ],
            generator=generator
        )

        train_dataloader = DataLoader(
            train_data, 
            batch_size=self.model_training_config.batch_size, 
            shuffle=True, collate_fn=self._collate_data_func
        )

        val_dataloader = DataLoader(
            test_data, 
            batch_size=self.model_training_config.batch_size, 
            shuffle=True, collate_fn=self._collate_data_func
        )

        return train_dataloader, val_dataloader
    
    def initialize_optimization_parameters(self, lr, weights = None) -> Tuple:
        """
        
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        return optimizer, loss_fn
    
    def __calculate_model_weights(self, input_ids, image_features, attn_mask, mode="train"):
        
        if self.model_training_config.device == "cuda":
            image_features = image_features.to(self.model_training_config.device) 
            input_ids = input_ids.to(self.model_training_config.device) 
            attn_mask = attn_mask.to(self.model_training_config.device)
            free_gpu_cache()
            self.model.zero_grad()
            logits = self.model(
                image_features=image_features, 
                input_ids=input_ids, 
                atten_mask=attn_mask
            )
            # Add padding to the target tensor to match the sequence length of the logits tensor
            pad_length = logits.size(1) - input_ids.size(1)
            target = torch.cat(
                [torch.full(
                    (input_ids.size(0), pad_length), 
                    -100, dtype=torch.long, 
                    device=input_ids.device
                ), input_ids], dim=1)

            loss = self.criterion(logits.view(-1, logits.size(-1)), target.view(-1))

            if mode == "train":
                loss.backward()
                self.optimizer.step()
            free_gpu_cache()
            return loss.item()        
    
    def _fit_model(self):
        self.model.train()
        train_epoch_loss = 0
        for image_features, input_ids, attn_mask in self._train_dataloader:
            train_epoch_loss += self.__calculate_model_weights(
                input_ids=input_ids,
                image_features=image_features, 
                attn_mask=attn_mask
            )
            free_gpu_cache()
        return train_epoch_loss/self.model_training_config.batch_size
    
    def _validate_model(self):
        self.model.eval()
        val_epoch_loss = 0
        with torch.no_grad():
            for image_features, input_ids, attn_mask in self._validation_dataloader:
                val_epoch_loss += self.__calculate_model_weights(
                    input_ids=input_ids,
                    image_features=image_features, 
                    attn_mask=attn_mask,
                    mode="validation"
            )
            free_gpu_cache()
        return val_epoch_loss/self.model_training_config.batch_size

    def get_predictions(self, dataloader):
        pass
