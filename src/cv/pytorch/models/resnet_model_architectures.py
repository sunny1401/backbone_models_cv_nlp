
RESNET_MODEL_ARCHITECTURES = dict(
    resnet18 = [
        ("BasicBlock", 1, 64, 2), 
        ("BasicBlock", 2, 128, 2), 
        ("BasicBlock", 2, 256, 2), 
        ("BasicBlock", 2, 512, 2)
    ],
    resnet34 = [
        ("BasicBlock", 1, 64, 3), 
        ("BasicBlock", 2, 128, 4), 
        ("BasicBlock", 2, 256, 6), 
        ("BasicBlock", 2, 512, 3)
    ],
    resnet50 = [
        ("BottleneckBlock", 1, 64, 3), 
        ("BottleneckBlock", 2, 128, 4), 
        ("BottleneckBlock", 2, 256, 6), 
        ("BottleneckBlock", 2, 512, 3)
    ],
    resnet101 = [
        ("BottleneckBlock", 1, 64, 3), 
        ("BottleneckBlock", 2, 128, 4), 
        ("BottleneckBlock", 2, 256, 23), 
        ("BottleneckBlock", 2, 512, 3)
    ],
    resnet152 = [
        ("BottleneckBlock", 1, 64, 3), 
        ("BottleneckBlock", 2, 128, 4), 
        ("BottleneckBlock", 2, 256, 36), 
        ("BottleneckBlock", 2, 512, 3)
    ],
)