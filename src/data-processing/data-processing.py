import os
import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import wiener
from scipy.ndimage import gaussian_filter
from skimage import img_as_float, img_as_uint
import time
import json

print("Hello World")

def verify_path(path):
    if not os.path.exists(path):
        print(f"Path does not exist: {path}", flush=True)
        exit(1)

def load_dicom_images(path, start_idx=0, batch_size=10):
    images = []
    patient_ids = []
    total_files = [os.path.join(root, file) for root, _, files in os.walk(path) for file in files if file.endswith('.dcm')]

    for filepath in total_files[start_idx:start_idx + batch_size]:
        dicom = pydicom.dcmread(filepath)
        image = dicom.pixel_array
        if image.dtype != np.uint16:
            image = image.astype(np.uint16)
        image = image.astype(np.float32)  # Convert to float32
        images.append(image)
        patient_ids.append(dicom.PatientID + "_" + os.path.basename(filepath).replace('.dcm', ''))

    return images, patient_ids

def apply_wiener(image, noise=None):
    try:
        image = img_as_float(image)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Scale to [0, 1]
        image -= 0.5  # Center to [-0.5, 0.5]
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
    result = clahe.apply(image)
    return result

def save_images(images, titles, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for image, title in zip(images, titles):
        output_path = os.path.join(output_dir, f'{title}.png')
        plt.imsave(output_path, image, cmap='gray')
        print(f"Processed image saved: {output_path}", flush=True)

def save_processed_images(processed_images, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(processed_images, f)
    print(f"Processed images dictionary saved at: {save_path}", flush=True)

def load_processed_images(load_path):
    if os.path.exists(load_path):
        with open(load_path, 'r') as f:
            return json.load(f)
    return {}

def process_image(image, patient_id, dataset_name, noise_levels, clip_limits, tile_grid_sizes, output_dir_base, first_image_dir=None):
    start_time = time.time()
    for clip_limit in clip_limits:
        for tile_grid_size in tile_grid_sizes:
            clahe_applied = apply_clahe(normalize_image(image), clip_limit=clip_limit, tile_grid_size=tile_grid_size)

            for noise in noise_levels:
                wiener_filtered = apply_wiener(clahe_applied, noise=noise)
                normalized_wiener = normalize_image(wiener_filtered)
                parameters = f"CL{clip_limit}_TG{tile_grid_size[0]}x{tile_grid_size[1]}_WN{noise}"
                output_dir = os.path.join(output_dir_base, dataset_name, parameters)
                save_images([normalized_wiener], [f"{patient_id}"], output_dir)

                if first_image_dir:
                    first_image_output_dir = os.path.join(first_image_dir, dataset_name, parameters)
                    save_images([normalized_wiener], [f"{patient_id}"], first_image_output_dir)
                    print(f"First processed image saved: {first_image_output_dir}/{patient_id}.png", flush=True)
                    first_image_dir = None

    end_time = time.time()
    print(f"Finished processing image {patient_id}. Time taken: {end_time - start_time} seconds", flush=True)

def process_dataset(path, dataset_name, noise_levels, clip_limits, tile_grid_sizes, output_dir_base, first_image_dir, processed_images_path, batch_size=10):
    processed_images = load_processed_images(processed_images_path)
    start_idx = 0
    while True:
        print(f"Processing batch starting at index {start_idx}", flush=True)
        images, patient_ids = load_dicom_images(path, start_idx=start_idx, batch_size=batch_size)
        if not images:
            print("No more images to process.", flush=True)
            break
        for image, patient_id in zip(images, patient_ids):
            patient_base_id = patient_id.split('_')[0]
            if patient_base_id not in processed_images:
                processed_images[patient_base_id] = set()
            if patient_id not in processed_images[patient_base_id]:
                process_image(image, patient_id, dataset_name, noise_levels, clip_limits, tile_grid_sizes, output_dir_base, first_image_dir if start_idx == 0 else None)
                processed_images[patient_base_id].add(patient_id)
        start_idx += batch_size
        save_processed_images(processed_images, processed_images_path)
        print(f"Batch starting at index {start_idx} processed.", flush=True)

def main():
    cmmd_path = "/data/bl70/validate/CMMD"
    cbis_ddsm_path = "/data/bl70/validate/CBIS-DDSM"

    verify_path(cmmd_path)
    verify_path(cbis_ddsm_path)

    noise_levels = [0.01, 0.1, 1.0]
    clip_limits = [2.0, 3.0, 5.0]
    tile_grid_sizes = [(8, 8), (16, 16)]

    output_dir_base = "/data/bl70/validate/ProcessedImages"
    first_image_dir = os.path.expanduser("~/CS5099-Code/data/1stImageTest")
    processed_images_path = os.path.expanduser("~/CS5099-Code/data/processed_images.json")

    print(f"First image directory: {first_image_dir}", flush=True)
    print(f"Processed images path: {processed_images_path}", flush=True)

    print("Starting dataset processing", flush=True)
    process_dataset(cmmd_path, "CMMD", noise_levels, clip_limits, tile_grid_sizes, output_dir_base, first_image_dir, processed_images_path, batch_size=10)
    process_dataset(cbis_ddsm_path, "CBIS-DDSM", noise_levels, clip_limits, tile_grid_sizes, output_dir_base, first_image_dir, processed_images_path, batch_size=10)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Script terminated by user.", flush=True)
    except Exception as e:
        print(f"An error occurred: {e}", flush=True)
