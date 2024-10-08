import torch.nn as nn
from typing import  Union
import torch.nn.functional as F
import torch


def get_activation(use_leaky_relu: bool, alpha: float = 0.01):
    if use_leaky_relu:
        return nn.LeakyReLU(negative_slope=alpha)
    else:
        return nn.ReLU()


class BatchConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 0,
        use_leaky_relu: bool = False,
        alpha_leaky_relu: float = 0.001,
        batch_norm_epsilon: float = 1e-05,
        batch_norm_momentum: float = 0.1,
        dropout_probability: float = 0.0,
        no_activation_added: bool = False
    ) -> None:
        super(BatchConvLayer, self).__init__()

        self._use_leaky_relu = use_leaky_relu
        self._alpha_leaky_relu = alpha_leaky_relu

        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(
                num_features=out_channels,
                eps=batch_norm_epsilon,
                momentum=batch_norm_momentum,
                affine=True,
                track_running_stats=True,
            ),
        ]
        if not no_activation_added:
            layers.append(get_activation(
                use_leaky_relu=self._use_leaky_relu, alpha = self._alpha_leaky_relu))
    
        layers.append(nn.Dropout(dropout_probability))
        self.conv = nn.Sequential(*layers)
            
    def forward(self, x):
        x = self.conv(x)

        return x
    

class MaxPoolLayer(nn.Module):
    def __init__(
        self, kernel_size: int, stride: int, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False
    ):
        super(MaxPoolLayer, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input_):
        if self.kernel_size > 1 and self.stride == 1:
            padding = self.kernel_size - 1
            zero_pad = torch.nn.ZeroPad2d((0, padding, 0, padding))
            input_ = zero_pad(input_)
        return F.max_pool2d(
            input_, self.kernel_size, self.stride, self.padding,
            self.dilation, self.ceil_mode, self.return_indices
        )