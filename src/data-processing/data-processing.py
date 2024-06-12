import os
import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import wiener
from scipy.ndimage import gaussian_filter, median_filter
from skimage import img_as_float, img_as_uint
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

print("Script is starting...", flush=True)

def verify_path(path):
    if os.path.exists(path):
        print(f"Path verified: {path}", flush=True)
    else:
        print(f"Path does not exist: {path}", flush=True)
        exit(1)

def load_dicom_images(path, limit=None):
    print(f"Loading DICOM images from: {path}", flush=True)
    images = []
    patient_ids = []
    count = 0
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
                count += 1
                if limit and count >= limit:
                    print(f"Loaded {count} images from {path}", flush=True)
                    return images, patient_ids
    print(f"Loaded {len(images)} images from {path}", flush=True)
    return images, patient_ids

def adaptive_fuzzy_median_filter(image, kernel_size=3, threshold=0.5):
    def fuzzy_membership_function(value, threshold):
        return 1 / (1 + np.exp(-10 * (value - threshold)))

    def adaptive_median_filter(window):
        median_value = np.median(window.cpu().numpy())
        deviations = np.abs(window.cpu().numpy() - median_value)
        max_deviation = np.max(deviations)

        if max_deviation == 0:
            return torch.tensor(median_value, device=device)

        fuzzy_memberships = fuzzy_membership_function(deviations / max_deviation, threshold)
        weighted_median = np.sum(window.cpu().numpy() * fuzzy_memberships) / np.sum(fuzzy_memberships)

        return torch.tensor(weighted_median, device=device)

    padded_image = torch.nn.functional.pad(image, (kernel_size//2,), mode='reflect')
    filtered_image = torch.zeros_like(image)

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
            noise = max(noise, 1e-10)  # Avoid zero noise
            return img_as_uint(wiener(smoothed_image, noise=noise))
        else:
            return img_as_uint(wiener(smoothed_image))
    except Exception as e:
        print(f"Error applying Wiener filter: {e}", flush=True)
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
        output_path = os.path.join(output_dir, f'{title}.png')
        plt.savefig(output_path)
        print(f"Saved image {output_path}", flush=True)
        plt.close()

def process_image(image, patient_id, noise_levels, kernel_sizes, thresholds, output_dir_base):
    start_time = time.time()
    image_tensor = torch.tensor(image).to(device)
    for kernel_size in kernel_sizes:
        for threshold in thresholds:
            fuzzy_filtered = adaptive_fuzzy_median_filter(image_tensor, kernel_size=kernel_size, threshold=threshold).to(device)
            normalized_fuzzy = normalize_image(fuzzy_filtered.cpu().numpy())
            clahe_fuzzy = apply_clahe(normalized_fuzzy)
            output_dir = os.path.join(output_dir_base, f"Fuzzy/KernalSize{kernel_size}")
            save_images([clahe_fuzzy], [f"Image_{patient_id}_Fuzzy_Kernel_{kernel_size}_Threshold_{threshold}"], output_dir)

    for noise in noise_levels:
        wiener_filtered = apply_wiener(image_tensor.cpu().numpy(), noise=noise)
        normalized_wiener = normalize_image(wiener_filtered)
        clahe_wiener = apply_clahe(normalized_wiener)
        output_dir = os.path.join(output_dir_base, f"Wiener/Noise{noise}")
        save_images([clahe_wiener], [f"Image_{patient_id}_Wiener_Noise_{noise}"], output_dir)

    end_time = time.time()
    print(f"Finished processing image {patient_id}. Time taken: {end_time - start_time} seconds", flush=True)

def main():
    cmmd_path = "/data/bl70/validate/CMMD"
    cbis_ddsm_path = "/data/bl70/validate/CBIS-DDSM"

    verify_path(cmmd_path)
    verify_path(cbis_ddsm_path)

    batch_size = 10
    noise_levels = [0.01, 0.1, 1.0]
    kernel_sizes = [3, 5]
    thresholds = [0.3, 0.5, 0.7]

    output_dir_base = "/data/bl70/validate/Processed"

    offset = 0
    while True:
        cmmd_images, cmmd_patient_ids = load_dicom_images(cmmd_path, limit=batch_size, offset=offset)
        if not cmmd_images:
            break

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for image, patient_id in zip(cmmd_images, cmmd_patient_ids):
                futures.append(executor.submit(process_image, image, patient_id, noise_levels, kernel_sizes, thresholds, output_dir_base))

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error occurred: {e}", flush=True)

        offset += batch_size

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Script terminated by user.", flush=True)
    except Exception as e:
        print(f"An error occurred: {e}", flush=True)
