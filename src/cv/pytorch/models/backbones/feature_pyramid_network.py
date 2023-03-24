import torch.nn as nn


class FPNFeatureExtractor(nn.Module):
    """
    Feature extractor for combining low resolution 
    high semantic content features with high resolution semantically weak features
    """
    def __init__(self, in_channels_list, out_channel=256, upsampling_mode="nearest") -> None:
        super(FPNFeatureExtractor, self).__init__()

        self.lateral_conv_layers = nn.ModuleList()
        self.output_conv_layers = nn.ModuleList()
        self._upsampling_mode=upsampling_mode

        for channel in in_channels_list:

            # layers to transform input feature maps to a consistent number of channels.
            # convolutions applied to each input feature map independently
            self.lateral_conv_layers.append(
                nn.Conv2d(
                    in_channels=channel, out_channels=out_channel,
                    kernel_size=1
                )
            )
            # layers for refining and smoothing output feature maps after combining them
            # use of 3x£ convolutions and padding=1, to maintain spatial resolution
            self.output_conv_layers.append(
                nn.Conv2d(
                in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1
                )
            )

    def _upsample_and_add(self, x, y):
        """
        Args:
            x, y: feature maps

        Function performs upsampling on x to maatch features from y, 
        using updsampling mode given. Default nearest neighbour interpolation.
        The function the performs element wise addition on x and y
        """
        _, _, h, w = y.size()
        return nn.functional.interpolate(x, size=(h, w), mode=self._upsampling_mode) + y
    
    def forward(self, feature_maps):

        # extract the last feature map in the provided feature map and 
        # apply last lateral convolution; to get the last_feature_map which is 
        # used in generation pyramid features
        last_feature_map = self.lateral_conv_layers[-1](feature_maps[-1])
        # Applying last output convolution layer in output_conv_layers list 
        # to last feature map and use this to create a new list of pyramid_features
        pyramid_features = [self.output_conv_layers[-1](last_feature_map)]

        for i in range(len(feature_maps) -2, -1, -1):
            # applied current lateral convolution to 
            # current feature map, producing new 
            # lateral map for generating the pyramid feature
            lateral_map = self.lateral_conv_layers[i](feature_maps[i])
            # updample latest feature map to size of lateral map
            upsample = self._upsample_and_add(latest_feature_map, lateral_map)
            output = self.output_conv_layers[i](upsample)
            pyramid_features.insert(0, output)
            latest_feature_map = upsample

        return pyramid_features
