import os
import pydicom
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

print("Pectoral Muscle Removal Script is starting...")

def verify_path(path):
    if os.path.exists(path):
        print(f"Path verified: {path}")
    else:
        print(f"Path does not exist: {path}")
        exit(1)

def load_dicom_images(path, limit=None):
    print(f"Loading DICOM images from: {path}")
    images = []
    filenames = []
    try:
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
                    filenames.append(file)
                    count += 1
                    if limit and count >= limit:
                        print(f"Loaded {count} images from {path}")
                        return images, filenames
        print(f"Loaded {len(images)} images from {path}")
    except KeyboardInterrupt:
        print("Image loading interrupted by user.")
    return images, filenames

def preprocess_image(image):
    # Apply Gaussian filter to smooth the image
    smoothed = gaussian_filter(image, sigma=1)
    # Apply CLAHE to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(smoothed)
    return enhanced

def segment_pectoral_muscle(image):
    # Convert to 8-bit image
    image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Convert to binary image using thresholding
    _, binary_image = cv2.threshold(image_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply morphological operations to clean up the segmentation
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find contours and select the largest one (pectoral muscle)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(image_8bit)
        cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        return mask
    else:
        return None

def remove_pectoral_muscle(image, mask):
    if mask is not None:
        result = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return result
    else:
        return image

def save_images(original_images, processed_images, filenames, output_dir):
    original_dir = os.path.join(output_dir, 'original')
    processed_dir = os.path.join(output_dir, 'processed')

    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    for original_image, processed_image, filename in zip(original_images, processed_images, filenames):
        original_output_path = os.path.join(original_dir, f"{filename}.png")
        processed_output_path = os.path.join(processed_dir, f"{filename}.png")

        plt.imsave(original_output_path, original_image, cmap='gray')
        plt.imsave(processed_output_path, processed_image, cmap='gray')

        print(f"Saved original image to {original_output_path}")
        print(f"Saved processed image to {processed_output_path}")

if __name__ == "__main__":
    try:
        cmmd_path = "/data/bl70/validate/CMMD"
        output_dir = "/app/pectoralremovaldata"

        verify_path(cmmd_path)

        # Process only 5 images for testing
        batch_size = 5
        images, filenames = load_dicom_images(cmmd_path, limit=batch_size)

        processed_images = []
        for i, image in enumerate(images):
            print(f"Processing image {i+1}/{len(images)}: {filenames[i]}")

            preprocessed_image = preprocess_image(image)
            mask = segment_pectoral_muscle(preprocessed_image)
            processed_image = remove_pectoral_muscle(preprocessed_image, mask)

            processed_images.append(processed_image)
            print(f"Finished processing image {i+1}/{len(images)}: {filenames[i]}")

        save_images(images, processed_images, filenames, output_dir)

    except KeyboardInterrupt:
        print("Script terminated by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
