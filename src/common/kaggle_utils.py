from dataclasses import dataclass
import subprocess
import pandas as pd
import os
import shutil
from pathlib import Path
import zipfile
from typing import Tuple


@dataclass
class KaggleDatasetCommands:

    list_all_competitions: str = "kaggle competitions list"
    survey_studies: str = "kaggle-survey-2022"
    transfer_learning_food_recognition: str = "transfer-learning-on-food-recognition"


class KaggleApi:

    KAGGLE_DATA_SAVE_LOCATION = os.path.join(
        str(Path(os.path.split(__file__)[0]).parent),
        "data"
    )
    def __init__(self, dataset_name) -> None:
        
        try:
            self._dataset_file_name = getattr(KaggleDatasetCommands, dataset_name)
        except AttributeError:
            raise ValueError(f"The dataset name {self._dataset_name} doesn't exist in Kaggle")
        self._save_path = os.path.join(self.KAGGLE_DATA_SAVE_LOCATION, f"{self._dataset_file_name}.zip")
        self._dataset_name = dataset_name
    
    @classmethod
    def list_all_kaggle_compeitions(cls) -> pd.DataFrame:

        file_name = os.path.join(cls.KAGGLE_DATA_SAVE_LOCATION, "kaggle_competition_list.txt")

        if not os.path.exists(file_name):

            with open(file_name, "w") as f:
                subprocess.call(KaggleDatasetCommands.list_all_competitions.split(" "), stdout=f)

        
        df = pd.read_fwf(file_name)
        df = df.iloc[1:, :]
        df.columns = ["competition_name", "deadline", "category", "reward", "team_count", "have_you_entered"]
        return df

    def download_kaggle_competition_dataset(self) -> str:

        """
        """

        if not os.path.exists(self._save_path):
            command = f"kaggle competitions download -c {self._dataset_file_name}".split(" ")
            
            subprocess.call(command)
            current_location = os.path.join(os.getcwd(), f"{self._dataset_file_name}.zip")

            if self._save_path != current_location:
                shutil.move(current_location, self._save_path)

        print(f"File downloaded and saved to:   {self._save_path}")
    
    def unzip_and_return_folder_details(self) -> Tuple[str]:

        directory_to_save = os.path.join(os.path.split(self._save_path)[0], self._dataset_file_name)

        if not os.path.exists(directory_to_save):
            if not os.path.exists(self._save_path):
                self.download_kaggle_competition_dataset()
            with zipfile.ZipFile(self._save_path, 'r') as zip_ref:
                zip_ref.extractall(directory_to_save)
            os.remove(self._save_path)

        files_extracted = os.listdir(directory_to_save)

        return directory_to_save, files_extracted

