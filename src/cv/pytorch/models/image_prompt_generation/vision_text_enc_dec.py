import os
import torch
import torch.nn as nn
from typing import Callable, Union, Tuple
import transformers

from src.utils.cuda import free_gpu_cache


class VisionTransformer(nn.Module):
    def __init__(
        self, 
        image_size: int, 
        embedding_dim : int = 768, 
        image_patches : int = 16,
        existing_model_path: str = 'google/vit-base-patch16-224'
    ):
        super().__init__()
        
        self._vit = transformers.ViTModel.from_pretrained(
            existing_model_path)
        self._linear = nn.Linear(self._vit.config.hidden_size, embedding_dim)
        self._num_patches = self._get_embedding_patches_required(
            image_size=image_size, image_patches=image_patches
        )

    def _get_embedding_patches_required(
        self, 
        image_size: Union[Tuple, int], 
        image_patches: int
    ) -> int:

        if isinstance(image_size, tuple):
            num_patches_horizontal = image_size[0] // image_patches
            num_patches_vertical = image_size[1] // image_patches
            num_patches = num_patches_horizontal * num_patches_vertical

        elif isinstance(image_size, int):
            num_patches = (image_size //image_patches) ** 2

        return num_patches

    def forward(self, x: torch.tensor) -> torch.tensor:
        # taking in the input tensor which is image enbeddings
        # the model returns a dictionary containing various outputs. 
        # last_hidden_state is a tensor of shape 
        # (batch_size, sequence_length, hidden_dim) containing embeddings for all tokens,
        # including the special [CLS] token and image patch token.
        x = self._vit(x)["last_hidden_state"]
        # this remove the [CLS] token from the embedding 
        # and we only need the embeddings of image patch tokens.
        x = x[:, 1: self._num_patches +1, :]
        x = self._linear(x)
        # (sequence_length, batch_size, embedding)
        x = x.permute(1, 0, 2)
        return x


class Gpt2Decoder(nn.Module):

    def __init__(
        self, 
        vocab_size: int = 50257, 
        existing_model_path: str = "gpt2"
    ):
        
        self._gpt2 = transformers.GPT2LMHeadModel.from_pretrained(
            existing_model_path)
        self._gpt2.resize_token_embeddings(vocab_size)

    def forward(
        self, 
        input_ids: torch.tensor, 
        image_embeddings: torch.tensor, 
        atten_mask: torch.tensor
    ) -> torch.tensor:
        """
        We aim to learn a mapping from 
        image embeddings to prompt text mapping.
        """
        # get input embeddings from gpt2
        input_embeds = self._gpt2.transformer.wte(input_ids)

        # changing image_embeddings to be same shape as input_embedding
        image_embeddings = image_embeddings.permute(1, 0, 2)

        input_embeds = torch.cat([image_embeddings, input_embeds], dim=1)
        # updating attention mask to account for image embeddings
        atten_mask = torch.cat(
            [
                torch.ones((
                    atten_mask.size(0), image_embeddings.size(1)
                ), dtype=torch.long, device=atten_mask.device),
                atten_mask
            ], dim=1
        )

        # passing the concatenated embeddings 
        # and updated atten_mask to the decoder
        outputs = self._gpt2(
            inputs_embeds=input_embeds, attention_mask=atten_mask)
        
        return outputs.logits


class VisionTextModel(nn.Module):

    def __init__(
        self, 
        text_model: Callable, 
        image_model: Callable
    ):
        
        self._text_decoder = text_model
        self._vision_encoder = image_model

    def forward(
        self, 
        image_features: torch.tensor, 
        input_ids: torch.tensor, 
        atten_mask: torch.tensor
    ) -> torch.tensor:
        image_embeddings = self._vision_encoder(image_features)
        logits = self._text_decoder(
            input_ids=input_ids, 
            image_embeddings=image_embeddings, 
            atten_mask=atten_mask
        )
        free_gpu_cache()
        return logits

