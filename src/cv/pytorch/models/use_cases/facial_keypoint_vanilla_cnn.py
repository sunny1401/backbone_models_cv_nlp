from src.cv.pytorch.models.vanillla_cnn import VanillaCNN
from typing import List, Tuple, Optional, Union, Dict
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class FacialKeypointVCNN(VanillaCNN):
    
    def __init__(
        self, 
        cnn_layers: int, 
        input_channel: List[int], 
        output_channels: List[int], 
        kernel_sizes: List[Tuple[int]],
        linear_layers: List[Tuple[int]],
        dropout_threshold: Union[List[float], float],
        dropout_addition_options: int = 1,
        initialize_model_from_given_parameters: bool = True,
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
        super().__init__()
        if dropout_addition_options not in {1, 2, 3, 4}:
            raise ValueError(
                "The value for arg dropout_addition_option is not recognized."
                "dropout_addition_option: int: presents a design option where "
                "four values are accepted:"
                "\n1 -> add dropout after each layer, "
                "requires dropout_threshold of type float or "
                "List where each value is the dropout value for each layer."
                "\n2 -> add dropout after each cnn layer, "
                "requires dropout_threshold of type float or "
                "List where each value is the dropout value for each CNN layer."
                "\n3 -> add dropout between linear layers"
                "requires dropout_threshold of type float or "
                "List where each value is the dropout value for each linear layer."
                "\n4 -> add dropout between cnn and linear layers"
                "requires dropout_threshold of type float."
            )

        dropout_option_map = {
            1: len(cnn_layers) + len(linear_layers),
            2: len(cnn_layers),
            3: len(linear_layers),
            4: 1
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

                if dropout_addition_options in {1, 2}:
                    if isinstance(dropout_threshold, List):
                        dropout_value = dropout_threshold[i]
                    else:
                        dropout_value = dropout_threshold

                    self.add_dropout(dropout_threshold=dropout_value)

            if dropout_addition_options == 4:
                if isinstance(dropout_threshold, List):
                    dropout_value = dropout_threshold[0]
                else:
                    dropout_value = dropout_threshold

                self.add_dropout(dropout_threshold=dropout_value)
                
            for i in range(len(linear_layers)):
                
                self.add_linear_layer(linear_layers[i])

                if isinstance(dropout_threshold, List):
                    if dropout_addition_options == 1:
                        dropout_value = dropout_threshold[len(cnn_layers + 1) + i]
                    elif dropout_addition_options == 3:
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
        