import os
import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import wiener
from scipy.ndimage import gaussian_filter
from skimage import img_as_float, img_as_uint

def load_dicom_images(path):
    images = []
    patient_ids = []
    try:
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('.dcm'):
                    filepath = os.path.join(root, file)
                    dicom = pydicom.dcmread(filepath)
                    image = dicom.pixel_array
                    if image.dtype != np.uint16:
                        image = image.astype(np.uint16)
                    images.append(image)
                    patient_ids.append(dicom.PatientID)
    except KeyboardInterrupt:
        print("Image loading interrupted by user.")
    return images, patient_ids

def adaptive_fuzzy_median_filter(image, kernel_size=3, threshold=0.5):
    def fuzzy_membership_function(value, threshold):
        return 1 / (1 + np.exp(-10 * (value - threshold)))

    def adaptive_median_filter(window):
        median_value = np.median(window)
        deviations = np.abs(window - median_value)
        max_deviation = np.max(deviations)
        
        if max_deviation == 0:
            return median_value
        
        fuzzy_memberships = fuzzy_membership_function(deviations / max_deviation, threshold)
        weighted_median = np.sum(window * fuzzy_memberships) / np.sum(fuzzy_memberships)
        
        return weighted_median

    padded_image = np.pad(image, pad_width=kernel_size//2, mode='reflect')
    filtered_image = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded_image[i:i+kernel_size, j:j+kernel_size]
            filtered_image[i, j] = adaptive_median_filter(window)
    
    return filtered_image

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

if __name__ == "__main__":
    try:
        cmmd_path = "/data/bl70/validate/CMMD"
        cbis_ddsm_path = "/data/bl70/validate/CBIS-DDSM"

        cmmd_images, cmmd_patient_ids = load_dicom_images(cmmd_path)
        cbis_ddsm_images, cbis_ddsm_patient_ids = load_dicom_images(cbis_ddsm_path)

        noise_levels = [0.01, 0.1, 1.0]
        kernel_sizes = [3, 5]
        thresholds = [0.3, 0.5, 0.7]

        for i, (cmmd_image, patient_id) in enumerate(zip(cmmd_images, cmmd_patient_ids)):
            print(f"Processing CMMD image {i+1}/{len(cmmd_images)}: PatientID={patient_id}")
            
            for kernel_size in kernel_sizes:
                for threshold in thresholds:
                    fuzzy_filtered = adaptive_fuzzy_median_filter(cmmd_image, kernel_size=kernel_size, threshold=threshold)
                    normalized_fuzzy = normalize_image(fuzzy_filtered)
                    clahe_fuzzy = apply_clahe(normalized_fuzzy)
                    output_dir = f"/data/bl70/validate/CMMD-Processed/Fuzzy/KernalSize{kernel_size}"
                    save_images([clahe_fuzzy], [f"CMMD_{patient_id}_Fuzzy_Kernel_{kernel_size}_Threshold_{threshold}"], output_dir)
            
            for noise in noise_levels:
                wiener_filtered = apply_wiener(cmmd_image, noise=noise)
                normalized_wiener = normalize_image(wiener_filtered)
                clahe_wiener = apply_clahe(normalized_wiener)
                output_dir = f"/data/bl70/validate/CMMD-Processed/Wiener/Noise{noise}"
                save_images([clahe_wiener], [f"CMMD_{patient_id}_Wiener_Noise_{noise}"], output_dir)

            print(f"Finished processing CMMD image {i+1}/{len(cmmd_images)}.")

        for i, (cbis_ddsm_image, patient_id) in enumerate(zip(cbis_ddsm_images, cbis_ddsm_patient_ids)):
            print(f"Processing CBIS-DDSM image {i+1}/{len(cbis_ddsm_images)}: PatientID={patient_id}")
            
            for kernel_size in kernel_sizes:
                for threshold in thresholds:
                    fuzzy_filtered = adaptive_fuzzy_median_filter(cbis_ddsm_image, kernel_size=kernel_size, threshold=threshold)
                    normalized_fuzzy = normalize_image(fuzzy_filtered)
                    clahe_fuzzy = apply_clahe(normalized_fuzzy)
                    output_dir = f"/data/bl70/validate/CBIS-DDSM-Processed/Fuzzy/KernalSize{kernel_size}"
                    save_images([clahe_fuzzy], [f"CBIS_DDSM_{patient_id}_Fuzzy_Kernel_{kernel_size}_Threshold_{threshold}"], output_dir)
            
            for noise in noise_levels:
                wiener_filtered = apply_wiener(cbis_ddsm_image, noise=noise)
                normalized_wiener = normalize_image(wiener_filtered)
                clahe_wiener = apply_clahe(normalized_wiener)
                output_dir = f"/data/bl70/validate/CBIS-DDSM-Processed/Wiener/Noise{noise}"
                save_images([clahe_wiener], [f"CBIS_DDSM_{patient_id}_Wiener_Noise_{noise}"], output_dir)

            print(f"Finished processing CBIS-DDSM image {i+1}/{len(cbis_ddsm_images)}.")

    except KeyboardInterrupt:
        print("Script terminated by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
