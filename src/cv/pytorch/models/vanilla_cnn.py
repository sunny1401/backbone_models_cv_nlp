import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import torch

torch.backends.cudnn.deterministic = True


class VanillaCNN(nn.Module):
    def __init__(
        self, 
        alpha_leaky_relu: float, 
        cnn_batch_norm_flag: bool,
        batch_norm_epsilon: float,
        batch_norm_momentum: float,
        linear_batch_norm_flag: float,
        pooling: Optional[Tuple[Tuple[str, int]]] = None
    ):
        
        super(VanillaCNN, self).__init__()
        self._net = list()
        self._cnn_call_iteration = 0
        self._dropout_call_iteration = 0
        self._linear_call_iteration = 0
        self._alpha_leaky_relu = alpha_leaky_relu
        self._add_batch_norm_after_cnn = cnn_batch_norm_flag
        self._add_batch_norm_after_linear = linear_batch_norm_flag
        self._batch_norm_epsilon = batch_norm_epsilon
        self._batch_norm_momentum = batch_norm_momentum
        self._pooling  = pooling

    def _load_model_from_disk(self, model_path: str):
        pass
    
        
    def single_cnn_activation_step(
        self, 
        input_channels: int, 
        output_channels: int, 
        kernel_size: Tuple[int, int],
        pool_type: str,
        pool_size: Tuple[int, int]
        
    ):
        
        """
        Args:
            input_channel: 
            output_channel: 
            kernel_size
        """

        # TODO - add stride, padding etc
        self._cnn_call_iteration += 1
        conv_layer = nn.Conv2d(
                in_channels=input_channels, 
                out_channels=output_channels, kernel_size=kernel_size)
        self._net.append((f"conv2d{self._cnn_call_iteration}", conv_layer))
        
        setattr(self, f"conv2d{self._cnn_call_iteration}", conv_layer)
        
        pool_type_dict = {
            "adaptive_max": nn.AdaptiveMaxPool2d,
            "avg": nn.AvgPool2d,
            "max": nn.MaxPool2d,
            "adaptive_avg": nn.AdaptiveAvgPool2d
        }
        pool_layer = pool_type_dict[pool_type](kernel_size=pool_size)
        self._net.append((f"{pool_type}_pool2d{self._cnn_call_iteration}", pool_layer))
        setattr(self, f"{pool_type}_pool2d{self._cnn_call_iteration}", pool_layer)

        if self._add_batch_norm_after_cnn:
            self.add_batch_norm_layer(num_channels=output_channels)
            

    def add_batch_norm_layer(self, num_channels):
        batch_norm_layer = nn.BatchNorm2d(
            num_features=num_channels, 
            eps=self._batch_norm_epsilon, 
            momentum=self._batch_norm_momentum, 
            affine=True, 
            track_running_stats=True
        )
        self._net.append((f"batch_norm2d{self._cnn_call_iteration}", batch_norm_layer))
        setattr(self, f"batch_norm2d{self._cnn_call_iteration}", batch_norm_layer)

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
    ):
        """
        """
        if not self._linear_call_iteration:
            linear_layer = nn.LazyLinear(out_features=out_features, bias=learn_additive_bias)
        else:
            linear_layer = nn.Linear(in_features=in_features, out_features=out_features, bias=learn_additive_bias)
        self._linear_call_iteration += 1
        self._net.append(
            (f"linear{self._linear_call_iteration}", linear_layer)
        )
        setattr(self, f"linear{self._linear_call_iteration}",linear_layer)

        if self._add_batch_norm_after_linear:
            self.add_batch_norm_layer(num_channels=out_features)



    def _get_conv_layer_output_shape(
        self, kernel, stride, padding, dilation
    ):

        pass
        
    def forward(self, sample):
        
        flattening_done = False
        for idx, item in enumerate(self._net):
            (call_type, callable_f)  = item
            
            flatten_condition = (
                not flattening_done and 
                (
                    "linear" in call_type or 
                    "dropout" in call_type and len(self._net) - idx in {1, 2}
                )
            )
            
            if flatten_condition:
                flattening_done = True
                
                sample = sample.view(sample.size(0), -1)

            sample = callable_f(sample)
            if "conv" in call_type or (
                "linear" in call_type and call_type[-1] != self._linear_call_iteration
            ):

                sample = F.leaky_relu(sample, negative_slope = self._alpha_leaky_relu)
        return sample