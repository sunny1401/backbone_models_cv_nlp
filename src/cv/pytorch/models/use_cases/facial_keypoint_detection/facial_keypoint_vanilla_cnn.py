from src.cv.pytorch.models.vanillla_cnn import VanillaCNN
from typing import List, Tuple, Optional, Union


class FacialKeypointVCNN(VanillaCNN):
    
    def __init__(
        self, 
        add_batch_norm: bool,
        batch_norm_epsilon: float,
        batch_norm_momentum: float,
        alpha_leaky_relu: float,
        cnn_layers: int, 
        input_channel: int, 
        output_channels: List[int], 
        kernel_sizes: List[Tuple[int]],
        linear_layers: int,
        dropout_addition_options: int = 1,
        dropout_threshold: Optional[Union[List[float], float]] = None,
        initialize_model_from_given_parameters: bool = True,
        load_from_file: bool= False,
        model_file_path: Optional[str] = None,

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
           alpha_leaky_relu=alpha_leaky_relu, 
           batch_norm_flag=add_batch_norm,
           batch_norm_epsilon=batch_norm_epsilon,
           batch_norm_momentum=batch_norm_momentum
        )
        if dropout_addition_options not in {0, 1, 2, 3, 4, 5, 6}:
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

        if not dropout_threshold and dropout_addition_options != 0:
            raise ValueError(
                "Please provide values for dropout threshold to add dropout layers")
        
        dropout_option_map = {
            1: (cnn_layers - 1) + (linear_layers - 1),
            2: cnn_layers -1,
            3: cnn_layers,
            4: linear_layers - 1,
            5: 1,
            6: linear_layers
        }

        if isinstance(dropout_threshold, List):
            if len(dropout_threshold) != dropout_option_map[dropout_addition_options]:
                raise ValueError(
                    "The number of dropout layers required "
                    "and the dropout thresholds provided do not match"
                )
            
        if not len(output_channels) >= linear_layers + cnn_layers:
            raise ValueError(
                "Please provide the output value "
                "for each layer cnn/linear visa output channel"
            )
        if not load_from_file and initialize_model_from_given_parameters:
            for i in range(cnn_layers):
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

            if dropout_addition_options in {3, 6}:
                if isinstance(dropout_threshold, List):
                    if dropout_addition_options == 3:
                        dropout_value = dropout_threshold[-1]
                    else:
                        dropout_value = dropout_threshold.pop(0)
                else:
                    dropout_value = dropout_threshold

                self.add_dropout(dropout_threshold=dropout_value)
                
            for i in range(linear_layers):
                
                current_layer_input = cnn_layers + i
                if not i:
                
                    self.add_linear_layer( 
                            out_features=output_channels[current_layer_input]
                        )
                else:
                    self.add_linear_layer(
                        in_features=output_channels[current_layer_input - 1], 
                        out_fatures=output_channels[current_layer_input]
                )

                if dropout_addition_options in {4, 6}:
                    if len(dropout_threshold):
                        if isinstance(dropout_threshold, List):
                            self.add_dropout(
                                dropout_threshold=dropout_threshold.pop(0)
                            )
                        else:
                            self.add_dropout(
                                dropout_threshold=dropout_option_map)
                    

        else:
            if not model_file_path:
                raise ValueError(
                    "Arg file_path should be provided with load_ffrom_file = True"
                )
            self._load_model_from_disk(model_file_path)
            # TODO: expand this route
        