# Data processing
import os
import pydicom
import numpy as np
import cv2
from skimage.filters import median
from skimage.restoration import denoise_wavelet
from skimage.exposure import equalize_hist, equalize_adapthist
from skimage import img_as_float
import matplotlib.pyplot as plt

def load_dicom_images(path, num_images=10):
    images = []
    for file in sorted(os.listdir(path))[:num_images]:
        if file.endswith('.dcm'):
            filepath = os.path.join(path, file)
            dicom = pydicom.dcmread(filepath)
            image = dicom.pixel_array
            images.append(image)
    return images

def adaptive_fuzzy_median_filter(image, kernel_size=3):
    """
    Advanced adaptive fuzzy median filtering.
    
    Parameters:
        image (ndarray): Input image.
        kernel_size (int): Size of the median filter kernel. Default is 3.
    
    Returns:
        ndarray: Filtered image.
    """
    
    def fuzzy_membership_function(value, threshold):
        """
        Simple fuzzy membership function.
        
        Parameters:
            value (float): Pixel intensity value.
            threshold (float): Threshold for fuzzy membership.
        
        Returns:
            float: Fuzzy membership value.
        """
        return 1 / (1 + np.exp(-10 * (value - threshold)))

    def adaptive_median_filter(window):
        """
        Apply adaptive median filtering using fuzzy logic.
        
        Parameters:
            window (ndarray): Local window of the image.
        
        Returns:
            float: Filtered pixel value.
        """
        median_value = np.median(window)
        deviations = np.abs(window - median_value)
        max_deviation = np.max(deviations)
        
        # Fuzzy membership based on deviation
        fuzzy_memberships = fuzzy_membership_function(deviations / max_deviation, 0.5)
        
        # Weighted median
        weighted_median = np.sum(window * fuzzy_memberships) / np.sum(fuzzy_memberships)
        
        return weighted_median

    # Pad the image to handle the borders
    padded_image = np.pad(image, pad_width=kernel_size//2, mode='reflect')
    filtered_image = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract the local window
            window = padded_image[i:i+kernel_size, j:j+kernel_size]
            filtered_image[i, j] = adaptive_median_filter(window)
    
    return filtered_image

def noise_removal(image):
    denoised_wavelet = denoise_wavelet(image, multichannel=False)
    denoised_median = median(denoised_wavelet, np.ones((3, 3)))
    return denoised_median

def normalize_image(image):
    norm_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return norm_image

def enhance_image(image):
    image_hist_eq = equalize_hist(image)
    image_adapthist_eq = equalize_adapthist(image)
    return image_hist_eq, image_adapthist_eq

def process_images(images):
    processed_images = []
    for image in images:
        fuzzy_filtered = adaptive_fuzzy_median_filter(image)
        noise_removed = noise_removal(fuzzy_filtered)
        normalized = normalize_image(noise_removed)
        hist_eq, adapthist_eq = enhance_image(normalized)
        processed_images.append((hist_eq, adapthist_eq))
    return processed_images

def display_images(images, titles):
    plt.figure(figsize=(15, 5))
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    cmmd_path = "/data/bl70/validate/CMMD"
    cbis_ddsm_path = "/data/bl70/validate/CBIS-DDSM"

    cmmd_images = load_dicom_images(cmmd_path)
    cbis_ddsm_images = load_dicom_images(cbis_ddsm_path)

    cmmd_processed = process_images(cmmd_images)
    cbis_ddsm_processed = process_images(cbis_ddsm_images)

    for original, (hist_eq, adapthist_eq) in zip(cmmd_images, cmmd_processed):
        display_images([original, hist_eq, adapthist_eq], ["Original", "Histogram Equalized", "Adaptive Histogram Equalized"])

    for original, (hist_eq, adapthist_eq) in zip(cbis_ddsm_images, cbis_ddsm_processed):
        display_images([original, hist_eq, adapthist_eq], ["Original", "Histogram Equalized", "Adaptive Histogram Equalized"])
