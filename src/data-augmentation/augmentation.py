import os
import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, shift, zoom, affine_transform
from skimage import img_as_float, img_as_uint
from skimage.transform import AffineTransform, warp

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

def augment_image(image):
    augmented_images = []
    titles = []
    
    # Original image
    augmented_images.append(image)
    titles.append("Original")

    # Horizontal flipping of the original image
    flipped_horizontally = np.fliplr(image)
    augmented_images.append(flipped_horizontally)
    titles.append("Original_Flipped_Horizontally")
    
    # Vertical flipping of the original image
    flipped_vertically = np.flipud(image)
    augmented_images.append(flipped_vertically)
    titles.append("Original_Flipped_Vertically")
    
    # Rotations
    angles = [45, 90, 135, 180, 225, 270]  # angles
    for angle in angles:
        rotated_image = rotate(image, angle, reshape=False)
        augmented_images.append(rotated_image)
        titles.append(f"Rotated_{angle}_degrees")

        # Apply flipping to rotated images only for 90 and 270 degrees
        if angle in [90, 270]:
            rotated_flipped_horizontally = np.fliplr(rotated_image)
            augmented_images.append(rotated_flipped_horizontally)
            titles.append(f"Rotated_{angle}_degrees_Flipped_Horizontally")
            
            rotated_flipped_vertically = np.flipud(rotated_image)
            augmented_images.append(rotated_flipped_vertically)
            titles.append(f"Rotated_{angle}_degrees_Flipped_Vertically")
    
    # Scaling
    scales = [0.9, 1.1]  # Scale down to 90%, scale up to 110%
    for scale in scales:
        scaled_image = zoom(image, scale)
        # Ensure the scaled image fits within the original frame size
        if scale > 1:
            center = tuple(np.array(scaled_image.shape) // 2)
            crop_start = tuple((np.array(scaled_image.shape) - np.array(image.shape)) // 2)
            crop_end = tuple(crop_start[i] + image.shape[i] for i in range(len(image.shape)))
            scaled_image = scaled_image[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1]]
        else:
            pad_width = [(0, image.shape[i] - scaled_image.shape[i]) for i in range(len(image.shape))]
            scaled_image = np.pad(scaled_image, pad_width, mode='constant', constant_values=0)
        augmented_images.append(scaled_image)
        titles.append(f"Scaled_{int(scale*100)}_percent")
    
    # Translation
    translations = [(-10, 10), (10, -10)]  # Example translations by 10% of width/height
    for t in translations:
        translated_image = shift(image, t)
        augmented_images.append(translated_image)
        titles.append(f"Translated_{t[0]}_x_{t[1]}_y")
    
    # Stretching
    stretch_factors = [1.2, 0.8]  # Stretch by 20%, shrink by 20%
    for factor in stretch_factors:
        transform = AffineTransform(scale=(factor, 1))
        stretched_image = warp(image, transform, mode='edge')
        augmented_images.append(stretched_image)
        titles.append(f"Stretched_{int(factor*100)}_percent")
    
    # Shearing
    shear_factors = [0.2, -0.2]  # Shear by 20% in each direction
    for shear in shear_factors:
        transform = AffineTransform(shear=shear)
        sheared_image = warp(image, transform, mode='edge')
        augmented_images.append(sheared_image)
        titles.append(f"Sheared_{int(shear*100)}_percent")
    
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
