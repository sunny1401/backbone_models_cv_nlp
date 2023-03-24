import torch.nn as nn
from typing import Optional, Dict, List, Union
import torch.nn.functional as F
from torch import nn


import torch.nn as nn
from typing import Optional, Dict
from torch import nn
from src.cv.pytorch.models.architecture_cutomisation import ConvLayer, get_activation


class ResnetBasicBlock(nn.Module):

    expansion: int = 1

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        backpropagation_relu_details: Optional[Dict] = None, 
        check_for_downsample: bool = False,
        dropout_probability: float = 0.0,
        stride=1,
        use_leaky_relu: bool = False,
        yolo_compatible: bool = False
    ) -> None:

        super(ResnetBasicBlock, self).__init__()
        backpropagation_relu_details = dict() if not backpropagation_relu_details else backpropagation_relu_details
        alpha_leaky_relu=backpropagation_relu_details.get("alpha_leaky_relu", 0.001)
        batch_norm_epsilon=backpropagation_relu_details.get("batch_norm_epsilon", 1e-05)
        batch_norm_momentum=backpropagation_relu_details.get("batch_norm_momentum", 0.1)
        use_leaky_relu = use_leaky_relu if not yolo_compatible else True
        dropout_probability = dropout_probability if not yolo_compatible else 0.0

        self.conv1 = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=stride,
            padding=1,
            use_leaky_relu=use_leaky_relu,
            backpropagation_relu=backpropagation_relu_details.get("backpropagation_relu", False),
            alpha_leaky_relu=alpha_leaky_relu,
            batch_norm_epsilon=batch_norm_epsilon,
            batch_norm_momentum=batch_norm_momentum,
            dropout_probability=dropout_probability
        )
        self.conv2 = ConvLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding=1,
            use_leaky_relu=use_leaky_relu,
            backpropagation_relu=False,
            alpha_leaky_relu=alpha_leaky_relu,
            batch_norm_epsilon=batch_norm_epsilon,
            batch_norm_momentum=batch_norm_momentum,
            dropout_probability=dropout_probability
            no_activation_added=True
        )


        self.downsample = None

        if check_for_downsample and (stride != 1 or in_channels != self.expansion * out_channels):
            self.downsample = nn.Sequential(
                ConvLayer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(1, 1),
                    stride=stride,
                    use_leaky_relu=False,
                    backpropagation_relu=False,
                    alpha_leaky_relu=alpha_leaky_relu,
                    batch_norm_epsilon=batch_norm_epsilon,
                    batch_norm_momentum=batch_norm_momentum,
                    dropout_probability=dropout_probability
                    no_activation_added=True
                )
            )

        self.activation = get_activation(use_leaky_relu=use_leaky_relu, alpha = alpha_leaky_relu)

    def forward(self, img):
        identity = img.clone()

        img = self.conv1(img)
        img = self.conv2(img)

        if self.downsample:
            identity = self.downsample(identity)

        img += identity

        img = self.activation(img)

        return img



class ResnetBottleneckBlock(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        backpropagation_relu_details: Optional[Dict] = None,
        check_for_downsample: bool = False,
        dropout_probability: float = 0.0,
        stride: int = 1,
        use_leaky_relu: bool = False,
    ) -> None:

        super(ResnetBottleneckBlock, self).__init__()

        backpropagation_relu_details = dict() if not backpropagation_relu_details else backpropagation_relu_details
        self._alpha_leaky_relu = backpropagation_relu_details.get("alpha_leaky_relu", 0.001)
        self._batch_norm_epsilon = backpropagation_relu_details.get("batch_norm_epsilon", 1e-05)
        self._batch_norm_momentum = backpropagation_relu_details.get("batch_norm_momentum", 0.1)
        self._use_leaky_relu = use_leaky_relu

        self.conv1 = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            use_leaky_relu=use_leaky_relu,
            alpha_leaky_relu=self._alpha_leaky_relu,
            batch_norm_epsilon=self._batch_norm_epsilon,
            batch_norm_momentum=self._batch_norm_momentum,
            dropout_probability=dropout_probability
        )

        self.conv2 = ConvLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=stride,
            padding=1,
            use_leaky_relu=use_leaky_relu,
            alpha_leaky_relu=self._alpha_leaky_relu,
            batch_norm_epsilon=self._batch_norm_epsilon,
            batch_norm_momentum=self._batch_norm_momentum,
            dropout_probability=dropout_probability
        )

        self.conv3 = ConvLayer(
            in_channels=out_channels,
            out_channels=out_channels * self.expansion,
            kernel_size=(1, 1),
            use_leaky_relu=False,
            no_activation_added=True,
            alpha_leaky_relu=self._alpha_leaky_relu,
            batch_norm_epsilon=self._batch_norm_epsilon,
            batch_norm_momentum=self._batch_norm_momentum,
            dropout_probability=dropout_probability
        )

        self.downsample = None

        if check_for_downsample and (stride != 1 or in_channels != self.expansion * out_channels):
            self.downsample = nn.Sequential(
                ConvLayer(
                    in_channels=in_channels,
                    out_channels=out_channels * self.expansion,
                    kernel_size=(1, 1),
                    stride=stride,
                    use_leaky_relu=False,
                    no_activation_added=True,
                    alpha_leaky_relu=self._alpha_leaky_relu,
                    batch_norm_epsilon=self._batch_norm_epsilon,
                    batch_norm_momentum=self._batch_norm_momentum,
                    dropout_probability=dropout_probability
                )
            )
        self.activation = get_activation(use_leaky_relu=self._use_leaky_relu, alpha = self._alpha_leaky_relu)

    def forward(self, img):
        identity = img.clone()

        img = self.conv1(img)
        img = self.conv2(img)
        img = self.conv3(img)

        if self.downsample:
            identity = self.downsample(identity)

        img += identity

        img = self.activation(img)

        return img


class VanillaResnet(nn.Module):

    def __init__(
        self, 
        device: str,
        in_channels: int, 
        num_classes: int, 
        resnet_stride_output_combination: List,
        alpha_leaky_relu: float=0.001, 
        batch_norm_epsilon: float = 1e-05,
        batch_norm_momentum: float = 0.1,
        cnn_batch_norm_flag: bool = True,
        linear_batch_norm_flag: bool = True,
        use_leaky_relu_in_resnet: bool = False,
        dropout_probability: Optional[Union[float, List[float]]] = 0.01,
        num_linear_layers: int = 1, 
        output_channels: Optional[List[int]] = None,
        add_dropout_to_linear_layers: bool = True,
    ):

        super(VanillaResnet, self).__init__()

        self._backpropagation_relu_dict = dict(
            alpha_leaky_relu=alpha_leaky_relu,
            batch_norm_epsilon=batch_norm_epsilon,
            batch_norm_momentum=batch_norm_momentum,
            cnn_batch_norm_flag=cnn_batch_norm_flag,
            linear_batch_norm_flag=linear_batch_norm_flag
        )
        self._device = device
        self._basic_block_count = 1
        self._bottleneck_block_count = 1
        self._required_input_channels = 64

        self.conv1 = ConvLayer(
            in_channels=in_channels,
            out_channels=self._required_input_channels,
            kernel_size=(7,7),
            stride=2,
            padding=3,
            batch_norm_epsilon=batch_norm_epsilon,
            batch_norm_momentum=batch_norm_momentum,
            use_leaky_relu=use_leaky_relu_in_resnet,
            alpha_leaky_relu=alpha_leaky_relu,
        )
        
        self.pool = nn.MaxPool2d(
            kernel_size=(3,3),
            stride=2,
            padding=1
        )

        self._num_classes = num_classes
        self.resnet_block = nn.ModuleList()
        cnn_dropout_probability = (
            dropout_probability[0] if isinstance(dropout_probability, List) else dropout_probability
        )
        for (block_type, stride, output, num_layers) in resnet_stride_output_combination:

            if block_type == "BasicBlock":
                self._basic_block_count += 1
                self.add_basic_block(
                    out_channels=output, 
                    stride=stride,
                    use_leaky_relu=use_leaky_relu_in_resnet,
                    num_layers=num_layers,
                    dropout_probability=cnn_dropout_probability
                )  

            elif block_type == "BottleneckBlock":
                self._bottleneck_block_count += 1
                self.add_bottleneck_block(
                    out_channels=output,
                    stride=stride,
                    use_leaky_relu=use_leaky_relu_in_resnet,
                    num_layers=num_layers,
                    dropout_probability=cnn_dropout_probability
                )

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        if num_linear_layers > 1:

            if output_channels != num_linear_layers - 1:

                raise ValueError(
                    "The final layer of the resnet is a fully connected network with "
                    f"output neurons = {self._num_classes}. "
                    "For other layers please provide relevant outputs"
                )
        if add_dropout_to_linear_layers:
            if isinstance(dropout_probability, List):
                if not len(dropout_probability) == num_linear_layers - 1:

                    raise ValueError("Please provide dropout for all layers save last")
            elif num_linear_layers > 1:
                dropout_probability = [dropout_probability] * (num_linear_layers - 1)
            else:
                dropout_probability = [dropout_probability]

        self.linear_layers = nn.ModuleList()
        for i in range(num_linear_layers - 1):

            network_layers = list()
            network_layers.append(
                nn.Linear(
                    in_features=self._required_input_channels, 
                    out_features=output_channels[i], 
                    bias=True
                )
            )
            network_layers.append(
                nn.BatchNorm1d(
                    num_features=output_channels[i], 
                    eps=batch_norm_epsilon, 
                    momentum=batch_norm_momentum, 
                    affine=True, 
                    track_running_stats=True
                )
            )
            network_layers.append(nn.ReLU())
            if add_dropout_to_linear_layers:
                network_layers.append(
                    nn.Dropout(dropout_probability[i])
                )

            linear = nn.Sequential(*network_layers)
            self._required_input_channels = output_channels[i]
            self.linear_layers.append(linear)

        activation = get_activation(use_leaky_relu=False)
        final_layers = [
            nn.Linear(
                in_features=self._required_input_channels, 
                out_features=self._num_classes, 
                bias=True
            ),
            nn.BatchNorm1d(
                num_features=self._num_classes, 
                eps=batch_norm_epsilon, 
                momentum=batch_norm_momentum, 
                affine=True, 
                track_running_stats=True
            ),
            activation
        ]
        if add_dropout_to_linear_layers:
            final_layers.append(nn.Dropout(dropout_probability[-1]))

        self.final_layers = nn.Sequential(*final_layers)

   
    def add_basic_block(
        self, 
        dropout_probability: float,
        out_channels: int, 
        num_layers: int, 
        stride: int, 
        use_leaky_relu: bool = True
    ):
        in_channels = self._required_input_channels
        check_for_downsample = False
        for i in range(num_layers):
            if i == 0:
                check_for_downsample = True

            basic_block = ResnetBasicBlock(
                check_for_downsample=check_for_downsample,
                in_channels=in_channels,
                out_channels=out_channels, 
                stride=stride,
                use_leaky_relu=use_leaky_relu,
                backpropagation_relu_details=self._backpropagation_relu_dict,
                dropout_probability=dropout_probability,
            ).to(self._device)
            # setattr(self, f"BasicBlock_conv{self._basic_block_count}x_{i+1}", basic_block)
            self.resnet_block.append(basic_block)
            in_channels = out_channels * ResnetBasicBlock.expansion
        self._required_input_channels = out_channels

    def add_bottleneck_block(
        self, 
        dropout_probability: float,
        out_channels: int, 
        num_layers: int, 
        stride: int, 
        use_leaky_relu: bool = True
    ):
        in_channels = self._required_input_channels
        check_for_downsample = False
        for i in range(num_layers):
            if i == 0:
                check_for_downsample = True
            bottleneck_block = ResnetBottleneckBlock(
                check_for_downsample=check_for_downsample,
                in_channels=in_channels,
                out_channels=out_channels, 
                stride=stride,
                use_leaky_relu=use_leaky_relu,
                backpropagation_relu_details=self._backpropagation_relu_dict,
                dropout_probability=dropout_probability
            ).to(self._device)
            # setattr(self, f"BottleneckBlock_conv{self._bottleneck_block_count}x_{i+1}", bottleneck_block)
            self.resnet_block.append(bottleneck_block)
            in_channels = out_channels * ResnetBasicBlock.expansion

        self._required_input_channels = out_channels
    
    def forward(self, x):
        x = self.conv1(x)
        
        for layer in self.resnet_block:
            x = layer(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        for layer in self.linear_layers:
            x = layer(x)

        x  = self.final_layers(x)
        return x
