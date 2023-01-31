from matplotlib import pyplot as plt
import numpy as np


def cross_validation_dataset():
    pass

def show_all_pictures_for_a_patient(dataset, idx):

    required_columns = [
            "cancer", "image_id", "patient_id", "laterality"
    ]
        
    df = dataset.dataset.image_labels.loc[:, required_columns]
    patient_id = df.loc[idx, required_columns].patient_id

    all_image_indices = df[df.patient_id == patient_id].index.tolist()
    length_data = len(all_image_indices)
    images = [dataset[i] for i in all_image_indices]

    num_columns = 4
    num_rows = round(length_data/num_columns)
    positions = range(1, length_data+1)
    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(f"Data for patient_id {patient_id}")
    for i in range(length_data):
        img = images[i]['image']
        ax = plt.subplot(num_rows, num_columns, positions[i])
        ax.imshow(np.squeeze(img))
        ax.set_title(
            f"lateral_view = {df.loc[all_image_indices[i], 'laterality']}," 
            f"\ncancer = {df.loc[all_image_indices[i], 'cancer']}"
        )
    plt.show()
    plt.close()
