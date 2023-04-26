import os
import polars as pl
import pandas as pd

import torch
from PIL import Image
from typing import Callable

from transformers import GPT2Tokenizer
import torchvision.transforms as transforms

from src.utils.constants import IMAGENET_MEAN_STD


class DataPreprocessor:

    def __init__(
        self, 
        image_size: int, 
        existing_tokenizer_path: str = "gpt2",
        max_length_tokenizer: int = 256
    ) -> None:
        self._transformations = self._get_transformations(
            image_size=image_size)
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            existing_tokenizer_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self._max_length_tokenizer = max_length_tokenizer

    def _get_transformations(self, image_size: int) -> Callable:

        transformations = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                IMAGENET_MEAN_STD[0],
                IMAGENET_MEAN_STD[1]
            )
        ])

        return transformations
    
    def transform_image(self, image_path:str) -> torch.tensor:

        image = Image.open(image_path).convert("RGB")
        image = self._transformations(image)
        return image


    def tokenize_prompt_text(self,  prompt):

        tokens = self.tokenizer(
            prompt, return_tensors="pt",
            padding=True, 
            truncation=True,
            max_length=self._max_length_tokenizer
        )

        prompt = tokens["input_ids"].squeeze(0)
        atten_mask = tokens["attention_mask"].squeeze(0)
        return prompt, atten_mask
    

class ChunkedDataframe:

    def __init__(
        self, 
        image_meta_df: pd.DataFrame,
        chunked_data_save_folder: str,
        image_batch_folder: str,
        training_config: Callable,
        step_size: int = 32000,
        tokenizer_path: str = "gpt2"

    ):

        self._batch_size = training_config.batch_size
        self._df = image_meta_df
        self._image_batch_folder = image_batch_folder
        self._step_size = step_size
        self._save_data_folder = chunked_data_save_folder
        self.preprocessor = DataPreprocessor(
            image_size=training_config.image_size,
            existing_tokenizer_path=tokenizer_path,
            max_length_tokenizer=training_config.max_length_tokenizer
        )

    def _chunk_data(self, start: int, end: int, db_number: int):

        indices = [i for i in range(start, end)]
        df = self._df[self._df.index.isin(indices)].reset_index(drop=True)
        batch_save_folder = os.path.join(self._image_batch_folder, db_number)
        os.makedirs(batch_save_folder, exist_ok=True)
        num_batches = df.shape[0] //self._batch_size

        image_path = df.file_path.tolist()

        prompts = df.prompt.tolist()

        for i in range(num_batches):
            file_location = os.path.join(batch_save_folder, f"batch_{i}.pth")
            batch_start = i * self._batch_size
            batch_end = (i + 1) * self._batch_size

            image_batch  = [
                self.preprocessor.transform_image(img_path)
                for img_path in image_path[batch_start: batch_end]
            ]

            tokenized_text = [
                self.preprocessor.tokenize_prompt_text(prompt) 
                for prompt in prompts[batch_start: batch_end]
            ]

            tokenized_prompt = [val[0] for val in tokenized_text]
            atten_mask = [val[1] for val in tokenized_text]

            torch.save({
                "images": image_batch, 
                "prompts": tokenized_prompt, 
                "attn_mask": atten_mask
            }, file_location)

    def _generate_processed_data_chunks(self, end_after: int = -1):
        
        start = 0
        end = start + self._step_size
        iter = 0
        while(end <= self._df.shape[0]):

            print(
                f"Generating chunk {iter} "
                f"which starts at {start} and ends at {end}"
            )

            self._chunk_data(start=start, end=end, db_number=iter)
            start += self._step_size
            end += self._step_size

            if end > self._df.shape[0]:
                end = self._df.shape[0]
            iter += 1
            if iter == end_after:
                break
