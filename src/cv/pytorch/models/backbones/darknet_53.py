from src.cv.pytorch.models.architecture_cutomisation import ConvLayer

from torch import nn


class DarknetResidualBlock(nn.Module):

    def __init__(self, channels=3) -> None:
        super(DarknetResidualBlock, self).__init__()

        self.residual = nn.Sequential(
            ConvLayer(
                in_channels=channels, 
                out_channels=channels//2, 
                kernel_size=1, 
                stride=1, 
                use_leaky_relu=True, 
            ),
            ConvLayer(
                in_channels=channels//2, 
                out_channels=channels, 
                kernel_size=3, 
                stride=1, 
                padding=1,
                use_leaky_relu=True, 
            ),
        )

    def forward(self, x):
        return x + self.residual(x)


class Darknet53(nn.Module):

    def __init__(
            self, 
            alpha_leaky_relu=0.1,
            batch_norm_epsilon=0.00001, 
            batch_norm_momentum=0.1, 
            dropout_probability=0.0

        ) -> None:
        super(Darknet53, self).__init__()

        self.layers = [

            ConvLayer(
                in_channels=3, 
                out_channels=32, 
                kernel_size=3, 
                stride=1, 
                padding=1,
                use_leaky_relu=True, 
                alpha_leaky_relu=alpha_leaky_relu,
                batch_norm_epsilon=batch_norm_epsilon,
                batch_norm_momentum=batch_norm_momentum,
                dropout_probability=dropout_probability
            ),
            ConvLayer(
                in_channels=32,
                out_channels=64, 
                kernel_size=3, 
                stride=2, 
                padding=1,
                use_leaky_relu=True, 
                alpha_leaky_relu=alpha_leaky_relu,
                batch_norm_epsilon=batch_norm_epsilon,
                batch_norm_momentum=batch_norm_momentum,
                dropout_probability=dropout_probability
            ),
            DarknetResidualBlock(64),
            ConvLayer(
                in_channels=64,
                out_channels=128, 
                kernel_size=3, 
                stride=2, 
                padding=1,
                use_leaky_relu=True, 
                alpha_leaky_relu=alpha_leaky_relu,
                batch_norm_epsilon=batch_norm_epsilon,
                batch_norm_momentum=batch_norm_momentum,
                dropout_probability=dropout_probability
            ),
            self._get_residual_layer_stack(128, 2),
            ConvLayer(
                in_channels=128,
                out_channels=256, 
                kernel_size=3, 
                stride=2, 
                padding=1,
                use_leaky_relu=True, 
                alpha_leaky_relu=alpha_leaky_relu,
                batch_norm_epsilon=batch_norm_epsilon,
                batch_norm_momentum=batch_norm_momentum,
                dropout_probability=dropout_probability
            ),
            self._get_residual_layer_stack(256, 8),
            ConvLayer(
                in_channels=256,
                out_channels=512, 
                kernel_size=3, 
                stride=2, 
                padding=1,
                use_leaky_relu=True, 
                alpha_leaky_relu=alpha_leaky_relu,
                batch_norm_epsilon=batch_norm_epsilon,
                batch_norm_momentum=batch_norm_momentum,
                dropout_probability=dropout_probability
            ),
            self._get_residual_layer_stack(512, 8),
            ConvLayer(
                in_channels=512,
                out_channels=1024, 
                kernel_size=3, 
                stride=2, 
                padding=1,
                use_leaky_relu=True, 
                alpha_leaky_relu=alpha_leaky_relu,
                batch_norm_epsilon=batch_norm_epsilon,
                batch_norm_momentum=batch_norm_momentum,
                dropout_probability=dropout_probability
            ),
            self._get_residual_layer_stack(1024, 4),

        ]

    def _get_residual_layer_stack(self, channels, num_repeat):
        layers = []

        for _ in range(num_repeat):
            layers.append(DarknetResidualBlock(channels))
        return nn.Sequential(*layers)
    

    def forward(self, x):
        self.layers = nn.Sequential(*self.layers)
        return self.layers(x)
    
