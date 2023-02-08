from dataclasses import dataclass
import torch
import numpy as np

@dataclass
class ModelDataConfig:
    dataset_size: int
    model_save_dir: str
    random_seed: int
    shuffle_dataset: bool 
    train_size: int
    validation_size: int
    show_dataset_plot: bool = True
    train_data_pct: float = 0.7

    def __init__(
        self, 
        dataset_size: int, 
        model_save_path: str,
        train_data_pct: float, 
        shuffle_dataset: bool = True,
        random_seed: int = 14,
        show_dataset_plots: bool = True
    ):

        """
        """
        if train_data_pct > 1:
            raise ValueError(
                "Value needs to be less than or equal to 1."
            )

        np.random.seed(random_seed)
        self.random_seed = random_seed
        self.dataset_size = dataset_size
        self.model_save_dir = model_save_path
        self.train_size = int(np.floor(train_data_pct * dataset_size))
        self.validation_size = dataset_size - self.train_size
        self.shuffle_dataset = shuffle_dataset
        self.show_dataset_plot = show_dataset_plots
        self.train_data_pct = train_data_pct
    

@dataclass(init=True)
class ModelTrainingConfig:
    learning_rate: float
    batch_size: int
    epochs: int
    add_batch_norm: bool = True
    alpha_leaky_relu: float = 0.1
    batch_norm_epsilon: float = 1e-05 
    batch_norm_momentum: float = 0.1
    device: str = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
