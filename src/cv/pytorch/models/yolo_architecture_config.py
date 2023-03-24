yolov3_architecture_definition = dict(
    backbone=[
        # Initial layers
        dict(filters=32, kernel_size=3, stride=1),
        dict(filters=64, kernel_size=3, stride=2),

        # Residual Blocks
        dict(repeat=1, layers=[
            dict(filters=32, kernel_size=1, stride=1), 
            dict(filters=64, kernel_size=3, stride=1),
        ]),
        dict(filters=128, kernel_size=3, stride=2),
        dict(repeat=2, layers=[
            dict(filters=64, kernel_size=1, stride=1), 
            dict(filters=128, kernel_size=3, stride=1),
        ]),
        dict(filters=256, kernel_size=3, stride=2),
        dict(repeat=8, layers=[
            dict(filters=128, kernel_size=1, stride=1),
            dict(filters=256, kernel_size=3, stride=1),
        ]),
        dict(filters=512, kernel_size=3, stride=2),
        dict(repeat=8, layers=[
            dict(filters=256, kernel_size=1, stride=1),
            dict(filters=512, kernel_size=3, stride=1),
        ]),
        dict(filters=1024, kernel_size=3, stride=2),
        dict(repeat=4, layers=[
            dict(filters=512, kernel_size=1, stride=1),
            dict(filters=1024, kernel_size=3, stride=1),
        ]),
    ],

    # # Detection layers
    # detection=[
    #     # Large objects
    #     dict(filters=512, kernel_size=1, stride=1),
    #     dict(filters=1024, kernel_size=3, stride=1),
    #     dict(filters=3 * (5 + num_classes), kernel_size=1, stride=1),

    #     # Medium objects
    #     dict(filters=256, kernel_size=1, stride=1),
    #     # Upsampling
    #     dict(filters=128, kernel_size=1, stride=1),
    #     dict(filters=256, kernel_size=3, stride=1),
    #     dict(filters=3 * (5 + num_classes), kernel_size=1, stride=1),

    #     # Small objects
    #     dict(filters=128, kernel_size=1, stride=1),
    #     # Upsampling
    #     dict(filters=64, kernel_size=1, stride=1),
    #     dict(filters=128, kernel_size=3, stride=1),
    #     dict(filters=3 * (5 + num_classes), kernel_size=1, stride=1),
    # ]
)
