import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import torch

torch.backends.cudnn.deterministic = True


class VanillaCNN(nn.Module):
    def __init__(
        self,  
        cnn_batch_norm_flag: bool,
        alpha_leaky_relu: float = 0.001,
        batch_norm_epsilon: float = 1e-05,
        batch_norm_momentum: float = 0.1,
        linear_batch_norm_flag: float = True,
    ):
        
        super(VanillaCNN, self).__init__()
        self._net = list()
        self._cnn_call_iteration = 0
        self._dropout_call_iteration = 0
        self._linear_call_iteration = 0
        self._separate_pool_layer = 0
        self._alpha_leaky_relu = alpha_leaky_relu
        self._add_batch_norm_after_cnn = cnn_batch_norm_flag
        self._add_batch_norm_after_linear = linear_batch_norm_flag
        self._batch_norm_epsilon = batch_norm_epsilon
        self._batch_norm_momentum = batch_norm_momentum

    def single_cnn_activation_step(
        self, 
        input_channels: int, 
        output_channels: int, 
        kernel_size: Tuple[int, int],
        pool_type: str = "max",
        pool_size: Tuple[int, int] = (2, 2),
        pool_stride: Optional[int] = None,
        pool_padding: int = 1,
        add_pooling: bool = True,
        add_batch_norm: bool = True,
        padding: int = 0,
        stride: int = 1,
        add_to_network: bool = True,
        non_relu_network: bool = False
    ):
        
        # TODO - comments and docstring
        # TODO - add flag for selecting relu
        self._cnn_call_iteration += 1
        layers = []
        layers.append(nn.Conv2d(
                in_channels=input_channels, 
                out_channels=output_channels, kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
        )
        if self._add_batch_norm_after_cnn or add_batch_norm:
            layers.append(
                self.get_batch_norm_layer(num_channels=output_channels)
            )

        if not non_relu_network:
            layers.append(nn.LeakyReLU(negative_slope=self._alpha_leaky_relu))
        
        if add_pooling:
            layers.append(
                self.get_pooling_layer(
                    pool_size=pool_size, pool_type=pool_type,
                    stride=pool_stride,
                    padding=pool_padding,
                )
            )

        final_conv_layer = nn.Sequential(*layers)
        if add_to_network:
            self._net.append((f"conv{self._cnn_call_iteration}", final_conv_layer))
            setattr(self, f"conv{self._cnn_call_iteration}", final_conv_layer)

        else:
            return final_conv_layer
            

    def get_pooling_layer(
        self, 
        pool_type: str, 
        pool_size: Tuple[int, int] = (2,2), 
        add_to_network: bool = True,
        stride: Optional[int] = None,
        padding: int = 0, 
    ):

        pool_type_dict = {
            "adaptive_max": nn.AdaptiveMaxPool2d,
            "avg": nn.AvgPool2d,
            "max": nn.MaxPool2d,
            "adaptive_avg": nn.AdaptiveAvgPool2d
        }

        if "adaptive" in pool_type:
            pool_layer = pool_type_dict[pool_type](pool_size)
        else:

            pool_layer = pool_type_dict[pool_type](kernel_size=pool_size, stride=stride, padding=padding)

        if add_to_network:
            self._separate_pool_layer += 1
            self._net.append((f"{pool_type}Pool2d{self._separate_pool_layer}", pool_layer))
            setattr(self, f"{pool_type}Pool2d{self._separate_pool_layer}", pool_layer)

        return pool_layer

    def get_batch_norm_layer(self, num_channels):
        batch_norm_layer = nn.BatchNorm2d(
            num_features=num_channels, 
            eps=self._batch_norm_epsilon, 
            momentum=self._batch_norm_momentum, 
            affine=True, 
            track_running_stats=True
        )
        return batch_norm_layer

    def add_dropout(self, dropout_threshold):
        self._dropout_call_iteration += 1
        dropout_layer = nn.Dropout(dropout_threshold)
        self._net.append((f"dropout{self._dropout_call_iteration}", dropout_layer))
        setattr(self, f"dropout{self._dropout_call_iteration}", dropout_layer)
        
    def add_linear_layer(
        self, 
        out_features, 
        learn_additive_bias=True, 
        in_features: Optional[int] = None, 
        add_relu: bool = True
    ):
        """
        """
        linear_layers = []
        if not self._linear_call_iteration and not in_features:
            linear_layers.append(nn.LazyLinear(out_features=out_features, bias=learn_additive_bias))
        else:
            linear_layers.append(nn.Linear(in_features=in_features, out_features=out_features, bias=learn_additive_bias))
        
        if add_relu:
            linear_layers.append(
                nn.LeakyReLU(negative_slope=self._alpha_leaky_relu)
            )
        if self._add_batch_norm_after_linear:
            linear_layers.append(
                self.get_batch_norm_layer(num_channels=out_features)
            )

        self._linear_call_iteration += 1
        linear_layer = nn.Sequential(*linear_layers)
        self._net.append(
            (f"linear{self._linear_call_iteration}", linear_layer)
        )
        setattr(self, f"linear{self._linear_call_iteration}",linear_layer)
        
    def forward(self, sample):
        
        flattening_done = False
        for _,  item in enumerate(self._net):
            (call_type, callable_f)  = item
            
            flatten_condition = (
                not flattening_done and "linear" in call_type
            )
            
            if flatten_condition:
                flattening_done = True
                
                sample = sample.view(sample.size(0), -1)

            sample = callable_f(sample)
            
        return sample