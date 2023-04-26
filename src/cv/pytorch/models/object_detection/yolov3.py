from src.cv.pytorch.models.backbones.darknet_53 import (
    Darknet53, DarknetResidualBlock
)
from src.cv.pytorch.models.backbones.feature_pyramid_network import FPN
from torch import nn
import torch


class Yolov3Darknet(Darknet53):

    def forward(self, x):

        feature_maps = []

        for layer in self.layers:
            x  = layer(x)
            if isinstance(layer, DarknetResidualBlock) and layer.residual[-1].out_channels in {256, 512, 1024}:
                feature_maps.append(x)

        return feature_maps[-3:]
    

class Yolov3DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors):

        """
        Anchors here are used by the model in guided generation of bounding boxes.

        """
        super(Yolov3DetectionHead, self).__init__()
        # num_anchors is the number of anchors that are used per grid
        # cell, while num_classes is the number of classes to be detected.

        self.prediction_layers = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            # The last layer is used to predict bounding box for 
            # each anchor along with class probabilities
            # Anchors has 5 attributed which need prediction
            # x,y coordinates of the center of object
            # width and height of the object
            # the objectness score  (i.e. probability that an object exists in the anchor)
            nn.Conv2d(512, num_anchors * (5 + num_classes), kernel_size=1),
        )

    def forward(self, x):
        return self.prediction_layers(x)
    

class Yolov3(nn.Module):

    def __init__(self, num_classes, num_anchors) -> None:
        super(Yolov3, self).__init__()

        self.darknet = Yolov3Darknet()

        # Run the Darknet53 forward function with a dummy input to get the feature maps
        dummy_input = torch.randn(1, 3, 416, 416)
        with torch.no_grad():
            feature_maps = self.darknet(dummy_input)
        
        # Initialize FPN and detection heads using the feature maps
        in_channels_list = [fm.size(1) for fm in feature_maps]
        self.fpn = FPN(in_channels_list=in_channels_list)

        self.detection_heads = nn.ModuleList([
            Yolov3DetectionHead(in_channels, num_classes, num_anchors)
            for in_channels in in_channels_list
        ])

    def forward(self, x):
        # Get feature maps from the Darknet53
        feature_maps = self.darknet(x)

        # Apply FPN to the feature maps
        fpn_outputs = self.fpn(feature_maps)

        # Apply detection heads to each FPN output
        detections = [
            head(fpn_output) for head, fpn_output in zip(self.detection_heads, fpn_outputs)
        ]

        return detections


