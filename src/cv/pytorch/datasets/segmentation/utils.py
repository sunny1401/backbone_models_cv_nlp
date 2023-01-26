from matplotlib import pyplot as plt
from torch.utils.data import Dataset



def show_image_label_side_by_side(
    dataset: Dataset, 
    idx: int, 
) -> int:
    """
    Function plots image and associated semantic labels on the image for the input idx
    """

    sample = dataset[idx]

    image, labels = sample["image"], sample["label"]
    # TODO: add color map
    # TODO: Add a way to show multiple rows of images
    _, ax = plt.subplots(1,2)
    ax[0].imshow(image.permute(1, 2, 0))
    ax[1].imshow(labels.permute(1, 2, 0))
    plt.pause(0.002)

    plt.show()

