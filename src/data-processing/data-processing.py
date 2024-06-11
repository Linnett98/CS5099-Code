import os
import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import wiener
from scipy.ndimage import gaussian_filter
from skimage import img_as_float, img_as_uint

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

def apply_wiener_custom(image, noise=0.01):
    image = img_as_float(image)
    image += 1e-10  # Add a very small constant to avoid zero variance regions
    wiener_filtered = wiener(image, noise=noise)
    return img_as_uint(wiener_filtered)

def apply_wiener(image, noise=None):
    try:
        image = img_as_float(image)
        image += 1e-10  # Add a small constant to avoid zero variance regions
        smoothed_image = gaussian_filter(image, sigma=1)
        if noise is not None:
            return img_as_uint(wiener(smoothed_image, noise=noise))
        else:
            return img_as_uint(wiener(smoothed_image))
    except Exception as e:
        print(f"Error applying Wiener filter: {e}")
        return image

def apply_gaussian_smoothing(image, sigma=1.0):
    return gaussian_filter(image, sigma=sigma)

def normalize_image(image):
    norm_image = (image - np.min(image)) * (65535.0 / (np.max(image) - np.min(image)))
    return norm_image.astype(np.uint16)

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    if image.dtype != np.uint16:
        image = np.uint16(image * 65535)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

def save_images(images, titles, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for image, title in zip(images, titles):
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'{title}.png'))
        plt.close()

def calculate_metrics(original, filtered):
    metrics = {}
    metrics['original_mean'] = np.mean(original)
    metrics['filtered_mean'] = np.mean(filtered)
    metrics['original_std'] = np.std(original)
    metrics['filtered_std'] = np.std(filtered)
    metrics['original_snr'] = metrics['original_mean'] / metrics['original_std']
    metrics['filtered_snr'] = metrics['filtered_mean'] / metrics['filtered_std']
    return metrics

def save_difference_image(original, filtered, title, output_dir):
    difference = np.abs(original - filtered)
    plt.figure()
    plt.imshow(difference, cmap='hot')
    plt.title('Difference Image')
    plt.colorbar()
    plt.axis('off')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{title}.png'))
    plt.close()

if __name__ == "__main__":
    cmmd_path = "/data/bl70/validate/CMMD"
    cbis_ddsm_path = "/data/bl70/validate/CBIS-DDSM"

    cmmd_images = load_dicom_images(cmmd_path, num_images=2)
    cbis_ddsm_images = load_dicom_images(cbis_ddsm_path, num_images=2)

    output_dir = "/home/bl70/CS5099-Code/data/metadata"

    noise_levels = [0.01, 0.1, 1.0]
    sigma_values = [0.5, 1.0, 2.0]
    clip_limits = [2.0, 3.0, 5.0]
    tile_grid_sizes = [(8, 8), (16, 16)]

    for i, cmmd_image in enumerate(cmmd_images):
        print(f"Processing CMMD image {i+1}:")
        save_images([cmmd_image], [f"Original_CMMD_Image_{i+1}"], output_dir)

        for noise in noise_levels:
            try:
                wiener_filtered = apply_wiener_custom(cmmd_image, noise=noise)
                save_images([wiener_filtered], [f"CMMD_Wiener_Filtered_{i+1}_Noise_{noise}"], output_dir)
                metrics = calculate_metrics(cmmd_image, wiener_filtered)
                print(f"Metrics for CMMD image {i+1} with Wiener filter (Noise {noise}): {metrics}")
                save_difference_image(cmmd_image, wiener_filtered, f"CMMD_Difference_{i+1}_Noise_{noise}", output_dir)
            except Exception as e:
                print(f"Error processing Wiener filter with noise {noise}: {e}")

        for sigma in sigma_values:
            smoothed = apply_gaussian_smoothing(cmmd_image, sigma=sigma)
            save_images([smoothed], [f"CMMD_Gaussian_Smoothed_{i+1}_Sigma_{sigma}"], output_dir)

        for clip_limit in clip_limits:
            for tile_grid_size in tile_grid_sizes:
                clahe_applied = apply_clahe(cmmd_image, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
                save_images([clahe_applied], [f"CMMD_CLAHE_{i+1}_Clip_{clip_limit}_Grid_{tile_grid_size}"], output_dir)

    for i, cbis_ddsm_image in enumerate(cbis_ddsm_images):
        print(f"Processing CBIS-DDSM image {i+1}:")
        save_images([cbis_ddsm_image], [f"Original_CBIS_DDSM_Image_{i+1}"], output_dir)

        for noise in noise_levels:
            try:
                wiener_filtered = apply_wiener_custom(cbis_ddsm_image, noise=noise)
                save_images([wiener_filtered], [f"CBIS_DDSM_Wiener_Filtered_{i+1}_Noise_{noise}"], output_dir)
                metrics = calculate_metrics(cbis_ddsm_image, wiener_filtered)
                print(f"Metrics for CBIS-DDSM image {i+1} with Wiener filter (Noise {noise}): {metrics}")
                save_difference_image(cbis_ddsm_image, wiener_filtered, f"CBIS_DDSM_Difference_{i+1}_Noise_{noise}", output_dir)
            except Exception as e:
                print(f"Error processing Wiener filter with noise {noise}: {e}")

        for sigma in sigma_values:
            smoothed = apply_gaussian_smoothing(cbis_ddsm_image, sigma=sigma)
            save_images([smoothed], [f"CBIS_DDSM_Gaussian_Smoothed_{i+1}_Sigma_{sigma}"], output_dir)

        for clip_limit in clip_limits:
            for tile_grid_size in tile_grid_sizes:
                clahe_applied = apply_clahe(cbis_ddsm_image, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
                save_images([clahe_applied], [f"CBIS_DDSM_CLAHE_{i+1}_Clip_{clip_limit}_Grid_{tile_grid_size}"], output_dir)
