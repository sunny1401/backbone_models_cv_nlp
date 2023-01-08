from dataclasses import dataclass
import torch


@dataclass
class ModelTrainingConfig:
    learning_rate: float
    batch_size: int
    epochs: int
    test_size: float
    train_size: float
    show_dataset_plot: True
    device: str = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, lr, batch_size, epochs, train_size, test_size):

        """
        """
        self.learning_rate = lr
        self.batch_size = batch_size
        self.epochs = epochs

        if train_size + test_size != 1:

            raise ValueError(
                "Train and test size should sum up 1."
                f"You provided train_size: {train_size} and "
                f"test_size: {test_size}"
            )
        self.test_size = test_size
        self.train_size = train_size