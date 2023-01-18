from src.cv.pytorch.models.vanillla_cnn import VanillaCNN
from src.cv.pytorch.models.configs import ModelTrainingConfig
from typing import List, Tuple, Optional, Union, Dict
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class FacialKeypointVCNN(VanillaCNN):
    
    def __init__(
        self, 
        cnn_layers: int, 
        input_channel: int, 
        output_channels: List[int], 
        kernel_sizes: List[Tuple[int]],
        linear_layers: List[Tuple[int]],
        dropout_threshold: Union[List[float], float],
        dropout_addition_options: int = 1,
        initialize_model_from_given_parameters: bool = True,
        leaky_relu_threshold: float = ModelTrainingConfig.alpha_leaky_relu,
        add_batch_norm: bool = ModelTrainingConfig.add_batch_norm,
        load_from_file: bool= False,
        file_path: Optional[str] = None
    ):
        """
        dropout_addition_option: int: presents a design option where 
        four values are accepted:
        1 -> add dropout after each layer
        2 -> add dropout after each cnn layer
        3 -> add dropout between linear layers
        4 -> add dropout between cnn and linear layers
        """

        super().__init__(
            leaky_relu_threshold = leaky_relu_threshold
            add_batch_norm = add_batch_norm
        )
        if dropout_addition_options not in {0, 1, 2, 3, 4, 5}:
            raise ValueError(
                "The value for arg dropout_addition_option is not recognized."
                "dropout_addition_option: int: presents a design option where "
                "seven values are accepted:"
                "\n0 -> no dropout added."
                "\n1 -> add dropout after each layer, "
                "requires dropout_threshold of type float or "
                "List where each value is the dropout value for each layer."
                "\n2 -> add dropout after each cnn layer, "
                "requires dropout_threshold of type float or "
                "List where each value is the dropout value for each CNN layer and for "
                "and linear layers and CNN layers "
                "\n3 -> add dropout between cnn and linear layers and all cnn layers; "
                "requires dropout threshold of type float or List of floats where "
                "each value is the dropout value for each ayer."
                "\n4 -> add dropout between linear layers"
                "requires dropout_threshold of type float or "
                "List where each value is the dropout value for each linear layer."
                "\n5 -> add dropout between cnn and linear layers"
                "requires dropout_threshold of type float."
                "\n6 -> add dropout between cnn and linear layers and all linear layers and "
                "requires dropout thrshold of type float or List of floats where "
                "each value is the dropout value for each layer."
            )

        dropout_option_map = {
            1: len(cnn_layers) + len(linear_layers),
            2: len(cnn_layers),
            3: len(cnn_layers) + 1,
            4: len(linear_layers),
            5: 1,
            6: len(linear_layers) + 1
        }

        if isinstance(dropout_threshold, List):
            if not len(dropout_threshold) != dropout_option_map[dropout_addition_options]:
                raise ValueError(
                    "The number of dropout layers required "
                    "and the dropout thresholds provided do not match"
                )
            
        if not load_from_file and initialize_model_from_given_parameters:
            for i in range(len(cnn_layers)):
                if i == 0:
                    input_channel = input_channel
                else:
                    input_channel = output_channels[i-1]


                self.single_cnn_activation_step(
                    input_channels=input_channel, 
                    output_channels=output_channels[i], 
                    kernel_size=kernel_sizes[i],
                    add_max_pool=True,
                    pool_size=(2,2),        
                )

                if dropout_addition_options in {1, 2, 3}:
                    if isinstance(dropout_threshold, List):
                        dropout_value = dropout_threshold[i]
                    else:
                        dropout_value = dropout_threshold

                    self.add_dropout(dropout_threshold=dropout_value)

            if dropout_addition_options in {3, 4, 6}:
                if isinstance(dropout_threshold, List):
                    dropout_value = dropout_threshold[-1]
                else:
                    dropout_value = dropout_threshold

                self.add_dropout(dropout_threshold=dropout_value)
                
            for i in range(len(linear_layers)):
                
                self.add_linear_layer(linear_layers[i])

                if dropout_addition_options in {4, 6}:

                    if isinstance(dropout_threshold, List):
                        if dropout_addition_options == 6:
                            dropout_value = dropout_threshold[len(cnn_layers + 1) + i]
                        elif dropout_addition_options == 4:
                            dropout_value = dropout_threshold[i]
                        
                    self.add_dropout(dropout_threshold=dropout_value)

        else:
            if not file_path:
                raise ValueError(
                    "Arg file_path should be provided with load_ffrom_file = True"
                )
            self._load_model_from_disk(file_path)
        
        
    def initialize_optimization_parameters(self, lr=0.0005) -> Dict:
        
        optimization_functions = dict()
        optimization_functions["criterion"] = nn.MSELoss()
        optimization_functions["optimizer"] = optim.Adam(
            self.parameters(), lr=lr
        )
        
        return optimization_functions
        