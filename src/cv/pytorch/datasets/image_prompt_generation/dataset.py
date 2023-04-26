import os

import torch
from torch.utils.data import Dataset


class ImagePromptDataset(Dataset):

    def __init__(self, data_location: str, tokenizer_length: int = 256):

        self._batch_data_location = data_location
        self._tokenizer_length = tokenizer_length
        self._image_path_dict = dict()

    def __len__(self):
        if not len(self._image_path_dict):
            iter = 0
            for curr_path, directory, file in os.walk(
                self._batch_data_location):
                self._image_path_dict[iter] = os.path.join(directory, curr_path, file)
                iter += 1

        return len(self._image_path_dict)

    def __getitem__(self, idx):

        batch_data = torch.load(self._image_path_dict[idx])
        image_features = batch_data["images"]
        input_ids = batch_data["prompts"]
        attention_mask = batch_data["attn_mask"]

        # padding the input ids and attention mask to be of same size
        padded_input_ids = []
        padded_attention_mask = []

        for i in range(len(input_ids)):
            pad_length = self._tokenizer_length - input_ids[i].size(0)
            padded_input_ids.append(
                torch.cat([input_ids[i], torch.zeros(pad_length, dtype=torch.long)])
            )
            padded_attention_mask.append(
                torch.cat([attention_mask[i], torch.zeros(pad_length, dtype=torch.long)])
            )

        # convert lists to tensors
        image_features = torch.stack(image_features)
        input_ids = torch.stack(padded_input_ids)
        attention_mask = torch.stack(padded_attention_mask)

        return image_features, input_ids, attention_mask
