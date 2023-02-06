import torch.nn as nn
from typing import Callable, Optional, Tuple, Union, Dict, List
import torch.nn.functional as F
from collections import OrderedDict
from src.cv.pytorch.models.vanilla_cnn import VanillaCNN


class ResnetBlock(VanillaCNN):

    
    def __init__(
        self, 
        input_shape: Tuple,
        output_shape: Tuple, 
        model_details: Dict, 
        num_layers: int, 
        stride: int, 
        resnet_type: str,
        batch_norm_momentum: float = 0.1,
        batch_norm_epsilon: float = 1e-05,
        alpha_leaky_relu: float = 0.01,
        use_leaky_relu: bool = False,
        downsample_function: Optional[Callable]= None,
    ):
        
        self._required_resnet_input_shape = 64

        stride_resnet_map = dict(
            resnet_18=[stride, 1],
            resnet_34=[stride, 1],
            resnet_50=[1, stride, 1],
            resnet_101=[1, stride, 1],
            lresnet_152=[1, stride, 1],
        )
        self._alpha_leaky_relu = alpha_leaky_relu
        self._batch_norm_epsilon = batch_norm_epsilon,
        self._batch_norm_momentum = batch_norm_momentum
        self._downsample_function = downsample_function
        self._layer_list = []
        self._use_leaky_relu = use_leaky_relu
        self.final_output_size = -1

        self.__add_resnet_block(
            num_layers=num_layers,
            stride_list=stride_resnet_map[resnet_type],
            input_size=input_shape,
            output_size=output_shape,
            layer_expansion=model_details["input_expansion"],
            kernel_size=model_details["kernel_size"],
            padding=model_details["padding"],
            resnet_type=resnet_type
        )
        
    def __add_resnet_block(
        self,
        num_layers: int, 
        stride_list: List[int], 
        input_size: int, 
        output_size: int, 
        layer_expansion: int, 
        kernel_size: Tuple, 
        padding: Tuple,
        resnet_type: str = "resnet_18"
    ):

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * len(num_layers)

        if isinstance(padding, int):
            padding = [padding] * len(num_layers)

        for i in range(num_layers):

            resnet_block = nn.Sequential(
                nn.conv2d(
                    input_size,
                    output_size,
                    kernel_size=kernel_size[i],
                    stride=stride_list[i],
                    padding=padding[i],
                    bias=False
                ),
                nn.BatchNorm2d(
                    output_size, 
                    eps=self._batch_norm_epsilon, 
                    momentum=self._batch_norm_momentum
                ),
            )
            self._layer_list.append(resnet_block)
            input_size = output_size
            if resnet_type in {"resnet_50", "resnet_101", "resnet_152"} and i == num_layers - 1:
                
                output_size = layer_expansion * output_size
            
            setattr(self, f"Basic{resnet_type.capitalize()}Block_{i+1}", resnet_block)
            self.final_output_size = output_size

    def forward(self, x):

        identity_x = x.clone()
        for idx, callable_fn in enumerate(self._layer_list):

            x = callable_fn(x)
            if idx != len(self._layer_list):
                if self._use_leaky_relu:
                    x = F.leaky_relu(x, negative_slope = self._alpha_leaky_relu)
                else:
                    x = F.relu(x)
        
        if self._downsample_function:
            identity_x = self._downsample_function(identity_x)

        x += identity_x
        if self._use_leaky_relu:
            x = F.leaky_relu(x, negative_slope = self._alpha_leaky_relu)
        else:
            x = F.relu(x)
        return x


class Resnet(VanillaCNN):

    kernel_input_out_stride_map = dict(
        resnet_18=dict(
            kernel_size=3,
            input_expansion=1,
            num_layers=dict(
                conv2=2,
                conv3=2,
                conv4=2,
                conv5=2
            ),
            padding=1,
            starting_input_size=64,
        ),
        resnet_34=dict(
            kernel_size=3,
            input_expansion=1,
            num_layers=dict(
                conv2=3,
                conv3=4,
                conv4=6,
                conv5=3
            ),
            padding=1,
            starting_input_size=64,
            
        ),
        resnet_50=dict(
            input_expansion=4,
            kernel_sizes=(1, 3, 1),  
            padding=(0,1,0),
            num_layers=dict(
                conv2=3,
                conv3=4,
                conv4=6,
                conv5=3
            ),
            starting_input_size=64,
        ),
        resnet_101=dict(
            input_expansion=4,
            kernel_sizes=(1, 3, 1),  
            num_layers=dict(
                conv2=3,
                conv3=4,
                conv4=23,
                conv5=3
            ),
            padding=(0,1,0),
            starting_input_size=64,
            
        ),
        resnet_152=dict(            
            input_expansion=4,
            num_layers=dict(
                conv2=3,
                conv3=8,
                conv4=36,
                conv5=3
            ),
            kernel_sizes=(1, 3, 1),  
            padding=(0,1,0),
            starting_input_size=64,
        )
    )
        
    def __init__(
        self,
        batch_norm_epsilon: float, 
        batch_norm_momentum: float,
        input_shape: Tuple[int, int],
        number_of_classes: Tuple[int, int],
        resnet_type: str = "resnet_18",
        use_leaky_relu_in_resnet: float = False,
        alpha_leaky_relu: float = 0.01,
        per_block_starting_stride: List = [1, 2, 2, 2]
    ):

        super().__init__(
           alpha_leaky_relu=alpha_leaky_relu, 
           cnn_batch_norm_flag=True,
           batch_norm_epsilon=batch_norm_epsilon,
           batch_norm_momentum=batch_norm_momentum,
           linear_batch_norm_flag=False
        )
        self._required_resnet_input_channels = 64
        self._use_leaky_relu = use_leaky_relu_in_resnet
        
        self.single_cnn_activation_step(
            input_channels=input_shape,
            output_channels=self._required_resnet_input_channels,
            kernel_size=(7,7),
            stride=2,
            padding=3,
            pool_type="max",
            pool_stride=2,
            pool_size=(3,3)
        )

        self._downsample_call = 0
        self._per_block_starting_stride = {
            f"conv{i + 2}": per_block_starting_stride[i]
            for i in range(len(self._per_block_starting_stride))
        }
        self._final_output_size = -1
        self._resent_block(resnet_type=resnet_type)
        self.get_pooling_layer(pool_type="avg", pool_size=(7,7), stride=1)
        self.add_linear_layer(
            in_features=self._final_output_size,
            out_features=number_of_classes
        )


    def _downsample_resnet_input(self, required_channel_size, given_starting_stride):

        self._downsample_call += 1
        downsample = nn.Sequential(
            nn.Conv2d(
                self._required_resnet_input_channels,
                required_channel_size,
                kernel_size=(1,1),
                stride=given_starting_stride,
                bias=False
            ),
            nn.BatchNorm2d(required_channel_size)
        )

        # self._net.append((f"Downsample2d{self._downsample_call}", downsample))
        # setattr(self, f"Downsample2d{self._downsample_call}", downsample)
        return downsample

    def _resent_block(self, resnet_type):

        model_details = self.kernel_input_out_stride_map[resnet_type]
        number_of_layers = model_details["num_layers"]

        starting_out_per_layer = [
            ("conv2", 64), ("conv3", 128), 
            ("conv4", 256), ("conv5", 512)
        ]
        
        for layer_name, starting_output in starting_out_per_layer:
            num_layers = number_of_layers[layer_name]
            downsample_condition = (
                self._per_block_starting_stride[layer_name] != 1 or 
                self._required_resnet_input_channels != starting_output
            )

            if downsample_condition:
                downsample = self._downsample_resnet_input(
                    required_channel_size=starting_output,
                    given_starting_stride=self._per_block_starting_stride[layer_name]
                )

                num_layers = num_layers - 1
            else:
                downsample = None

            resnet = ResnetBlock(
                input_shape=self._required_resnet_input_channels,
                output_shape=starting_output,
                model_details=model_details,
                num_layers=num_layers,
                stride=self._per_block_starting_stride[layer_name],
                resnet_type=resnet_type,
                use_leaky_relu=self._use_leaky_relu,
                downsample_function=downsample
            )
            self._net.append((f"{resnet_type}{layer_name}Block"), resnet)
            setattr(
                self, 
                f"{resnet_type}{layer_name}Block", resnet
            )
            self._final_output_size = resnet.final_output_size

    def forward(self, x):

        for (layer_name, callable_fn) in self._net:

            if "linear" in layer_name:
                x = x.reshape(x.shape[0], -1)

            x = callable_fn(x)

        return x

