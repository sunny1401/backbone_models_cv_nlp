from matplotlib import pyplot as plt
import numpy as np
from typing import List, Dict

def multi_view_image_keypoints(data: List[Dict], number_of_columns: int = 3):
    """
    TODO
    """
    length = len(data)
    number_of_rows = length//number_of_columns

    if (length%number_of_columns) != 0:
        number_of_rows += 1

    positions = range(1, length+1)
    plt.figure(figsize=(length, length))
    for i in range(length):
        img = data[i]['image']
        plt.subplot(number_of_rows, number_of_columns, positions[i])
        plt.imshow(img)
        plt.scatter(
            data[i]["facial_landmarks"][:, 0], 
            data[i]["facial_landmarks"][:, 1], s=20, c='r', marker="*")
    plt.show()
    plt.close()


def show_key_points_on_images(
    dataset, 
    idx, 
):
    """
    Function plots image and associated keypoint on the image for the input idx
    """

    sample = dataset[idx]
    plt.figure()
    plt.imshow(sample["image"], cmap='gray')
    plt.scatter(
        sample["facial_landmarks"][:, 0], 
        sample["facial_landmarks"][:, 1], s=20, c='r', marker="*")

    plt.pause(0.002)

    plt.show()