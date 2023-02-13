import torch.nn as nn
from typing import Optional, Dict, List, Union
import torch.nn.functional as F
from torch import nn


class ResnetBasicBlock(nn.Module):

    expansion: int = 1

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        backpropagation_relu_details: Optional[Dict] = None, 
        check_for_downsample: bool = False,
        stride=1,
        use_leaky_relu: bool = False
    ) -> None:

        super(ResnetBasicBlock, self).__init__()
        backpropagation_relu_details = dict() if not backpropagation_relu_details else backpropagation_relu_details
        self._alpha_leaky_relu=backpropagation_relu_details.get("alpha_leaky_relu", 0.001), 
        self._cnn_batch_norm_flag=backpropagation_relu_details.get("cnn_batch_norm_flag", True),
        self._batch_norm_epsilon=backpropagation_relu_details.get("batch_norm_epsilon", 1e-05),
        self._batch_norm_momentum=backpropagation_relu_details.get("batch_norm_momentum", 0.1),
        self._linear_batch_norm_flag=backpropagation_relu_details.get("linear_batch_norm_flag", True)
        self._use_leaky_relu = use_leaky_relu

        if self._use_leaky_relu:
            relu = nn.LeakyReLU(negative_slope=self._alpha_leaky_relu)
        else:
            relu = nn.ReLU()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3,3),
                stride=stride, 
                padding=1,
            ),
            nn.BatchNorm2d(
                num_features=out_channels, 
                eps=self._batch_norm_epsilon, 
                momentum=self._batch_norm_momentum, 
                affine=True, 
                track_running_stats=True
            ),
            relu
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3,3),
                padding=1,
            ),
            nn.BatchNorm2d(
                num_features=out_channels, 
                eps=self._batch_norm_epsilon, 
                momentum=self._batch_norm_momentum, 
                affine=True, 
                track_running_stats=True
            )
        )


        self.downsample = None

        if check_for_downsample and (stride != 1 or in_channels != self.expansion*out_channels):

            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(1,1),
                    stride=stride,
                ),
                nn.BatchNorm2d(
                    num_features=out_channels, 
                    eps=self._batch_norm_epsilon, 
                    momentum=self._batch_norm_momentum, 
                    affine=True, 
                    track_running_stats=True
                )
            )
            
        
    def forward(self, img):
        identity = img.clone()

        img = self.conv1(img)
        img = self.conv2(img)

        if self.downsample:
            identity = self.downsample(identity)
        img += identity
        if self._use_leaky_relu:
            img = F.leaky_relu(img, negative_slope=self._alpha_leaky_relu)
        else:
            img = F.relu(img)
        return img


class ResnetBottleneckBlock(nn.Module):

    expansion = 4

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        backpropagation_relu_details: Optional[Dict] = None, 
        check_for_downsample: bool = False,
        stride :int =1,
        use_leaky_relu: bool = False,
    ) -> None:

        super(ResnetBottleneckBlock, self).__init__()

        backpropagation_relu_details = dict() if not backpropagation_relu_details else backpropagation_relu_details
        self._alpha_leaky_relu=backpropagation_relu_details.get("alpha_leaky_relu", 0.001), 
        self._cnn_batch_norm_flag=backpropagation_relu_details.get("cnn_batch_norm_flag", True),
        self._batch_norm_epsilon=backpropagation_relu_details.get("batch_norm_epsilon", 1e-05),
        self._batch_norm_momentum=backpropagation_relu_details.get("batch_norm_momentum", 0.1),
        self._linear_batch_norm_flag=backpropagation_relu_details.get("linear_batch_norm_flag", True)
        self._use_leaky_relu = use_leaky_relu

        self._use_leaky_relu = use_leaky_relu

        if self._use_leaky_relu:
            relu = nn.LeakyReLU(negative_slope=self._alpha_leaky_relu)
        else:
            relu = nn.ReLU()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1,1),
            ),
            nn.BatchNorm2d(
                num_features=out_channels, 
                eps=self._batch_norm_epsilon, 
                momentum=self._batch_norm_momentum, 
                affine=True, 
                track_running_stats=True
            ),
            relu
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3,3),
                stride=stride,
                padding=1
            ),
            nn.BatchNorm2d(
                num_features=out_channels, 
                eps=self._batch_norm_epsilon, 
                momentum=self._batch_norm_momentum, 
                affine=True, 
                track_running_stats=True
            ),
            relu
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1,1),
            ),
            nn.BatchNorm2d(
                num_features=out_channels, 
                eps=self._batch_norm_epsilon, 
                momentum=self._batch_norm_momentum, 
                affine=True, 
                track_running_stats=True
            ),
            relu
        )

        self.downsample = None

        if check_for_downsample and (stride != 1 or in_channels != self.expansion*out_channels):
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(1,1),
                    stride=stride,
                ),
                nn.BatchNorm2d(
                    num_features=out_channels, 
                    eps=self._batch_norm_epsilon, 
                    momentum=self._batch_norm_momentum, 
                    affine=True, 
                    track_running_stats=True
                )
            )

    def forward(self, img):
        identity = img.clone()

        img = self.conv1(img)
        img = self.conv2(img)
        
        if self.downsample:
            identity = self.downsample(identity)
        img += identity
        if self._use_leaky_relu:
            img = F.leaky_relu(img, negative_slope=self._alpha_leaky_relu)
        else:
            img = F.relu(img)
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
        dropout_threshold: Optional[Union[float, List[float]]] = 0.01,
        num_linear_layers: int = 1, 
        output_channels: Optional[List[int]] = None,
        add_dropout_to_linear_layers: bool = True,
    ):

        super(VanillaResnet, self).__init__()

        # TODO - add dropout support within resnet
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

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=self._required_input_channels,
                kernel_size=(7,7),
                stride=2,
                padding=3,
            ),
            nn.BatchNorm2d(
                num_features=self._required_input_channels,
                eps=batch_norm_epsilon, 
                momentum=batch_norm_momentum, 
                affine=True, 
                track_running_stats=True
            ),
            nn.MaxPool2d(
                kernel_size=(3,3),
                stride=2,
                padding=1
            )

        )
        self._num_classes = num_classes
        self._resnet_block = []
        for (block_type, stride, output, num_layers) in resnet_stride_output_combination:

            if block_type == "BasicBlock":
                self._basic_block_count += 1
                self.add_basic_block(
                    out_channels=output, 
                    stride=stride,
                    use_leaky_relu=use_leaky_relu_in_resnet,
                    num_layers = num_layers
                )  

            elif block_type == "BottleneckBlock":
                self._bottleneck_block_count += 1
                self.add_bottleneck_block(
                    out_channels=output,
                    stride=stride,
                    use_leaky_relu=use_leaky_relu_in_resnet,
                    num_layers = num_layers
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
            if isinstance(dropout_threshold, List):
                if not len(dropout_threshold) == num_linear_layers - 1:

                    raise ValueError("Please provide dropout for all layers save last")
            elif num_linear_layers > 1:
                dropout_threshold = [dropout_threshold] * (num_linear_layers - 1)
            else:
                dropout_threshold = [dropout_threshold]

        for i in range(num_linear_layers - 1):

            network_layers = []
            network_layers.append(
                nn.Linear(
                    in_features=self._required_input_channels, 
                    out_features=output_channels[i], 
                    bias=True
                )
            )
            network_layers.append(
                nn.BatchNorm2d(
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
                    nn.Dropout(dropout_threshold[i])
                )

            linear = nn.Sequential(*network_layers)
            self._required_input_channels = output_channels[i]

            setattr(self, f"linear_layer_{i}", linear)
            self._resnet_block.append(("linear", linear))
        
        final_layers = [
            nn.Linear(
                in_features=self._required_input_channels, 
                out_features=self._num_classes, 
                bias=True
            ),
            nn.BatchNorm2d(
                num_features=self._num_classes, 
                eps=batch_norm_epsilon, 
                momentum=batch_norm_momentum, 
                affine=True, 
                track_running_stats=True
            ),
            nn.ReLU()
        ]
        if add_dropout_to_linear_layers:
            final_layers.append(
                nn.Dropout(dropout_threshold[-1])
            )

        self.final_layer = nn.Sequential(*final_layers)

   
    def add_basic_block(
        self, 
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
                backpropagation_relu_details=self._backpropagation_relu_dict
            ).to(self._device)
            setattr(self, f"BasicBlock_conv{self._basic_block_count}x_{i+1}", basic_block)
            self._resnet_block.append(("BBlock", basic_block))
            in_channels = out_channels * ResnetBasicBlock.expansion
        self._required_input_channels = out_channels

    def add_bottleneck_block(
        self, 
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
                backpropagation_relu_details=self._backpropagation_relu_dict
            ).to(self._device)
            setattr(self, f"BottleneckBlock_conv{self._bottleneck_block_count}x_{i+1}", bottleneck_block)
            self._resnet_block.append(("BBlock", bottleneck_block))
            in_channels = out_channels * ResnetBasicBlock.expansion

        self._required_input_channels = out_channels
    
    def forward(self, x):
        x = self.conv1(x)
        first_linear_occurence = False
        for layer_name, layer in self._resnet_block:
            if not first_linear_occurence and layer_name == "linear":
                x = self.avg_pool(x)
                x = x.view(x.size(0), -1)
                first_linear_occurence = True

            x = layer(x)

        if not first_linear_occurence:
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            first_linear_occurence = True

        x = self.final_layer(x)
        return x
