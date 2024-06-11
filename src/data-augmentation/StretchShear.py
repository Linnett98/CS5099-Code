import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform

def load_dicom_images(path, num_images=2):
    images = []
    for file in sorted(os.listdir(path))[:num_images]:
        if file.endswith('.dcm'):
            filepath = os.path.join(path, file)
            dicom = pydicom.dcmread(filepath)
            image = dicom.pixel_array
            if image.dtype != np.uint16:
                image = image.astype(np.uint16)
            images.append(image)
    return images

def center_image(image):
    non_zero_indices = np.nonzero(image)
    if non_zero_indices[0].size == 0 or non_zero_indices[1].size == 0:
        return image  # Image is completely black

    min_y, max_y = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
    min_x, max_x = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
    cropped_image = image[min_y:max_y+1, min_x:max_x+1]

    centered_image = np.zeros_like(image)
    start_y = (image.shape[0] - cropped_image.shape[0]) // 2
    start_x = (image.shape[1] - cropped_image.shape[1]) // 2
    centered_image[start_y:start_y+cropped_image.shape[0], start_x:start_x+cropped_image.shape[1]] = cropped_image

    return centered_image

def apply_shear(image, shear_factor):
    rows, cols = image.shape
    affine_matrix = np.array([[1, shear_factor, 0],
                              [0, 1, 0],
                              [0, 0, 1]])
    # Calculate the amount of padding needed to keep the image in the frame
    padding_width = int(abs(shear_factor) * rows)
    padded_image = np.pad(image, ((0, 0), (padding_width, padding_width)), mode='constant', constant_values=0)
    offset = np.dot(affine_matrix, [padded_image.shape[1] / 2, padded_image.shape[0] / 2, 1])[:2]
    sheared_image = affine_transform(padded_image, affine_matrix[:2, :2], offset=offset - [padded_image.shape[1] / 2, padded_image.shape[0] / 2], mode='constant', cval=0.0)
    
    # Crop the image back to its original size
    start_x = padding_width
    end_x = start_x + cols
    sheared_image = sheared_image[:, start_x:end_x]
    
    return sheared_image

def augment_image(image):
    augmented_images = []
    titles = []

    # Original image
    augmented_images.append(image)
    titles.append("Original")

    # Stretching
    stretch_factors = [1.2, 0.8]  # Stretch by 20%, shrink by 20%
    for factor in stretch_factors:
        rows, cols = image.shape
        if factor > 1:
            stretch_cols = int(cols * factor)
            if stretch_cols % 2 != 0:
                stretch_cols += 1
            padding_width = (stretch_cols - cols) // 2
            transform_matrix = np.array([[factor, 0, 0], [0, 1, 0], [0, 0, 1]])
            stretched_image = affine_transform(image, transform_matrix[:2, :2], offset=0, mode='constant', cval=0.0)
            stretched_image = stretched_image[:, padding_width:padding_width+cols]
        else:
            transform_matrix = np.array([[factor, 0, 0], [0, 1, 0], [0, 0, 1]])
            temp_image = affine_transform(image, transform_matrix[:2, :2], offset=0, mode='constant', cval=0.0)
            padding_width = (cols - temp_image.shape[1]) // 2
            stretched_image = np.zeros_like(image)
            stretched_image[:, padding_width:padding_width + temp_image.shape[1]] = temp_image
        augmented_images.append(stretched_image)
        titles.append(f"Stretched_{int(factor*100)}_percent")

    # Shearing
    shear_factors = [-0.2, 0.2]  # Example shear factors
    for shear_factor in shear_factors:
        if shear_factor < 0:
            centered_image = center_image(image)
            sheared_image = apply_shear(centered_image, shear_factor)
        else:
            sheared_image = apply_shear(image, shear_factor)
        augmented_images.append(sheared_image)
        titles.append(f"Sheared_{shear_factor*100:.0f}_percent")

    return augmented_images, titles

def save_images(images, titles, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    for image, title in zip(images, titles):
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
        filename = f"{prefix}_{title}.png".replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

if __name__ == "__main__":
    cmmd_path = "/data/bl70/validate/CMMD"
    cbis_ddsm_path = "/data/bl70/validate/CBIS-DDSM"

    cmmd_images = load_dicom_images(cmmd_path, num_images=2)
    cbis_ddsm_images = load_dicom_images(cbis_ddsm_path, num_images=2)

    output_dir = "/home/bl70/CS5099-Code/data/augmented"

    # Augment and save CMMD images
    for i, cmmd_image in enumerate(cmmd_images):
        print(f"Augmenting CMMD image {i+1}:")
        augmented_images, titles = augment_image(cmmd_image)
        save_images(augmented_images, titles, output_dir, f"CMMD_Augmented_{i+1}")

    # Augment and save CBIS-DDSM images
    for i, cbis_ddsm_image in enumerate(cbis_ddsm_images):
        print(f"Augmenting CBIS-DDSM image {i+1}:")
        augmented_images, titles = augment_image(cbis_ddsm_image)
        save_images(augmented_images, titles, output_dir, f"CBIS_DDSM_Augmented_{i+1}")
