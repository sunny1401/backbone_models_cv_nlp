import torch.nn as nn
from typing import Optional, Tuple, Dict, List, Union
import torch.nn.functional as F
from src.cv.pytorch.models.vanilla_cnn import VanillaCNN


class ResnetBasicBlock(VanillaCNN):

    expansion: int = 1

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        backpropagation_relu_details: Optional[Dict] = None, 
        stride=1,
        use_leaky_relu: bool = False
    ) -> None:

        backpropagation_relu_details = dict() if not backpropagation_relu_details else backpropagation_relu_details
        super(ResnetBasicBlock, self).__init__(
            alpha_leaky_relu=backpropagation_relu_details.get("alpha_leaky_relu", 0.001), 
            cnn_batch_norm_flag=backpropagation_relu_details.get("cnn_batch_norm_flag", True),
            batch_norm_epsilon=backpropagation_relu_details.get("batch_norm_epsilon", 1e-05),
            batch_norm_momentum=backpropagation_relu_details.get("batch_norm_momentum", 0.1),
            linear_batch_norm_flag=backpropagation_relu_details.get("linear_batch_norm_flag", True)
        )
        self._use_leaky_relu = use_leaky_relu
        
        self.single_cnn_activation_step(
            input_channels=in_channels,
            output_channels=out_channels, 
            kernel_size=(3,3),
            stride=stride,
            add_pooling=False,
        )

        self.single_cnn_activation_step(
            input_channels=out_channels,
            output_channels=out_channels, 
            kernel_size=(1,1),
            add_pooling=False,
            non_relu_network=True
        )

        self.downsample_identity = nn.Sequential()

        if stride != 1 or in_channels != self.expansion*out_channels:
            self.downsample_identity = self.single_cnn_activation_step(
                input_channels=in_channels,
                output_channels=out_channels*self.expansion, 
                kernel_size=(1,1),
                stride=stride,
                add_pooling=False,
            )
        
    def forward(self, img):

        identity = img.clone()
        for _, callable in self._net:

            img = callable(img)

        img += self.downsample_identity(identity)
        if self._use_leaky_relu:
            F.leaky_relu(negative_slope=self._alpha_leaky_relu, inplace=True)

        else:
            F.relu()

        return img


class ResnetBottleneckBlock(VanillaCNN):

    expansion = 4

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        backpropagation_relu_details: Optional[Dict] = None, 
        stride :int =1,
        use_leaky_relu: bool = False,
    ) -> None:

        backpropagation_relu_details = dict() if not backpropagation_relu_details else backpropagation_relu_details
        super(ResnetBottleneckBlock, self).__init__(
            alpha_leaky_relu=backpropagation_relu_details.get("alpha_leaky_relu", 0.001), 
            cnn_batch_norm_flag=backpropagation_relu_details.get("cnn_batch_norm_flag", True),
            batch_norm_epsilon=backpropagation_relu_details.get("batch_norm_epsilon", 1e-05),
            batch_norm_momentum=backpropagation_relu_details.get("batch_norm_momentum", 0.1),
            linear_batch_norm_flag=backpropagation_relu_details.get("linear_batch_norm_flag", True)
        )
        self._use_leaky_relu = use_leaky_relu

        self.single_cnn_activation_step(
            input_channels=in_channels,
            output_channels=out_channels, 
            kernel_size=(1,1),
            add_pooling=False,
        )

        self.single_cnn_activation_step(
            input_channels=out_channels,
            output_channels=out_channels, 
            kernel_size=(3,3),
            stride=stride,
            add_pooling=False,
        )        

        self.single_cnn_activation_step(
            input_channels=in_channels,
            output_channels=out_channels, 
            kernel_size=(1,1),
            add_pooling=False,
            add_to_network=False
        )

        self.downsample_identity = nn.Sequential()

        if stride != 1 or in_channels != self.expansion*out_channels:
            self.downsample_identity = self.single_cnn_activation_step(
                input_channels=in_channels,
                output_channels=out_channels*self.expansion, 
                kernel_size=(1,1),
                stride=stride,
                add_pooling=False,
            )
        

    def forward(self, img):

        identity = img.clone()
        for _, callable in self._net:

            img = callable(img)

        img += self.downsample_identity(identity)
        if self._use_leaky_relu:
            F.leaky_relu(negative_slope=self._alpha_leaky_relu, inplace=True)

        else:
            F.relu()

        return img


class VanillaResnet(VanillaCNN):

    def __init__(
        self, 
        in_channels: int, 
        num_classes: int, 
        initialize_cnn: bool, 
        resnet_stride_output_combination: List,
        alpha_leaky_relu: float=0.001, 
        batch_norm_epsilon: float = 1e-05,
        batch_norm_momentum: float = 0.1,
        cnn_batch_norm_flag: bool = True,
        linear_batch_norm_flag: bool = True,
        use_relu_in_resnet: bool = False
    ):

        # TODO - add dropout support within resnet
        super(VanillaResnet, self).__init__(
            alpha_leaky_relu=alpha_leaky_relu,
            batch_norm_epsilon=batch_norm_epsilon,
            batch_norm_momentum=batch_norm_momentum,
            cnn_batch_norm_flag=cnn_batch_norm_flag,
            linear_batch_norm_flag=linear_batch_norm_flag
        )

        self._required_input_channels = 64
        if initialize_cnn:
            self.single_cnn_activation_step(
                input_channels=in_channels,
                output_channels=self._required_input_channels,
                kernel_size=(7,7),
                stride=2,
                padding=3,
                pool_type="max",
                pool_stride=2,
                pool_size=(3,3)
            )
        self._num_classes = num_classes
        for (block_type, stride, output, num_layers) in resnet_stride_output_combination:

            if block_type == "BasicBlock":
              self.add_basic_block(
                out_channels=output, 
                stride=stride,
                use_leaky_relu=use_relu_in_resnet,
                num_layers = num_layers
            )  

            elif block_type == "BottleneckBlock":
                self.add_bottleneck_block(
                    out_channels=output,
                    stride=stride,
                    use_leaky_relu=use_relu_in_resnet,
                    num_layers = num_layers
                )

   
    def add_basic_block(
        self, 
        out_channels: int, 
        num_layers: int, 
        stride: int, 
        use_leaky_relu: bool = True
    ):

        for _ in range(num_layers):
            self._net.append(
                ("BasicBlock", ResnetBasicBlock(
                    in_channels=self._required_input_channels,
                    out_channels=out_channels, 
                    stride=stride,
                    use_leaky_relu=use_leaky_relu
                )
            ))
            self._required_input_channels = out_channels * ResnetBasicBlock.expansion

    def add_bottleneck_block(
        self, 
        out_channels: int, 
        num_layers: int, 
        stride: int, 
        use_leaky_relu: bool = True
    ):
        for _ in range(num_layers):
            self._net.append(
                ("BottleneckBlock", ResnetBottleneckBlock(
                    in_channels=self._required_input_channels,
                    out_channels=out_channels, 
                    stride=stride,
                    use_leaky_relu=use_leaky_relu
                )
            ))
            self._required_input_channels = out_channels * ResnetBottleneckBlock.expansion
    
    def wrap_up_network(
        self, 
        dropout_threshold: Union[float, List[float]],
        num_linear_layers: int = 1, 
        output_channels: Optional[List[int]] = None,
        add_dropout_to_layers: bool = True,
    ):

        if num_linear_layers > 1:

            if output_channels != num_linear_layers - 1:

                raise ValueError(
                    "The final layer of the resnet is a fully connected network with "
                    f"output neurons = {self._num_classes}. "
                    "For other layers please provide relevant outputs"
                )
        if add_dropout_to_layers and isinstance(dropout_threshold, List):
            if not len(dropout_threshold) == num_linear_layers - 1:

                raise ValueError("Please provide dropout for all layers save last")

        for i in range(num_linear_layers):
            if i == num_linear_layers[-1]:
                self._add_batch_norm_after_linear = False
                self.add_linear_layer(
                    in_features=self._required_input_channels,
                    out_features=self._num_classes,
                    add_relu=False
                )
            else:

                self.add_linear_layer(
                    in_features=self._required_input_channels,
                    out_features=output_channels[i],
                    add_relu=True
                )
                self._required_input_channels = output_channels[i]
                if add_dropout_to_layers:
                    if isinstance(dropout_threshold, List):
                        self.add_dropout(
                            dropout_threshold=dropout_threshold[i]
                        )
                    else:
                        self.add_dropout(dropout_threshold=dropout_threshold)
