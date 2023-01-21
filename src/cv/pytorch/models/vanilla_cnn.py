import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import torch

torch.backends.cudnn.deterministic = True


class VanillaCNN(nn.Module):
    def __init__(
        self, 
        alpha_leaky_relu: float, 
        batch_norm_flag: bool,
        batch_norm_epsilon: float,
        batch_norm_momentum: float
    ):
        
        super(VanillaCNN, self).__init__()
        self._net = list()
        self._cnn_call_iteration = 0
        self._dropout_call_iteration = 0
        self._linear_call_iteration = 0
        self._alpha_leaky_relu = alpha_leaky_relu
        self._batch_norm_flag = batch_norm_flag
        self._batch_norm_epsilon = batch_norm_epsilon
        self._batch_norm_momentum = batch_norm_momentum
        
    def _load_model_from_disk(self, model_path: str):
        pass
    
        
    def single_cnn_activation_step(
        self, 
        input_channels: int, 
        output_channels: int, 
        kernel_size: Tuple[int, int],
        add_max_pool: bool = False,
        pool_size: Optional[Tuple[int, int]] = None,
        
    ):
        
        """
        Args:
            input_channel: 
            output_channel: 
            kernel_size
        """
        
        # TODO - add validation on input
        # TODO - count number of GPU and use pipeline for more than one
        # TODO - add stride, padding etc
        self._cnn_call_iteration += 1
        conv_layer = nn.Conv2d(
                in_channels=input_channels, 
                out_channels=output_channels, kernel_size=kernel_size)
        self._net.append((f"conv2d{self._cnn_call_iteration}", conv_layer))
        
        setattr(self, f"conv2d{self._cnn_call_iteration}", conv_layer)
        
        if add_max_pool:
            if not pool_size:
                 raise ValueError("The pool size cannot be zero ")
            pool_layer = nn.MaxPool2d(kernel_size=pool_size)
            self._net.append((f"max_pool2d{self._cnn_call_iteration}", pool_layer))
            setattr(self, f"max_pool2d{self._cnn_call_iteration}", pool_layer)
            
            
    def add_dropout(self, dropout_threshold):
        self._dropout_call_iteration += 1
        dropout_layer = nn.Dropout(dropout_threshold)
        self._net.append((f"dropout{self._dropout_call_iteration}", dropout_layer))
        setattr(self, f"dropout{self._dropout_call_iteration}", dropout_layer)
        
    def add_linear_layer(self, out_features, learn_additive_bias=True, in_features: Optional[int] = None):
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
            if "conv" in call_type:
                
                sample = F.leaky_relu(sample, negative_slope = self._alpha_leaky_relu)
                if self._batch_norm_flag:
                    # inp is shape (N, C, H, W)
                    n_channels = sample.shape[1]
                    running_mu = torch.zeros(n_channels).to(sample.get_device()) # zeros are fine for first training iter
                    running_std = torch.ones(n_channels).to(sample.get_device()) # ones are fine for first training iter
                    sample = F.batch_norm(
                        sample, 
                        running_mu, 
                        running_std, 
                        training=True, 
                        momentum=self._batch_norm_momentum, 
                        eps=self._batch_norm_epsilon
                    )
        return sample