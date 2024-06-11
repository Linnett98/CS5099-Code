import os
import pydicom
import matplotlib.pyplot as plt

def load_and_extract_metadata(path, output_dir, num_images=2):
    count = 0
    for file in sorted(os.listdir(path)):
        if file.endswith('.dcm') and count < num_images:
            filepath = os.path.join(path, file)
            dicom = pydicom.dcmread(filepath)

            # Extract metadata
            patient_id = dicom.PatientID
            patient_name = dicom.get("PatientName", "Unknown")
            patient_birth_date = dicom.get("PatientBirthDate", "Unknown")
            patient_sex = dicom.get("PatientSex", "Unknown")
            study_instance_uid = dicom.get("StudyInstanceUID", "Unknown")
            study_date = dicom.get("StudyDate", "Unknown")
            series_instance_uid = dicom.get("SeriesInstanceUID", "Unknown")
            series_number = dicom.get("SeriesNumber", "Unknown")
            manufacturer = dicom.get("Manufacturer", "Unknown")
            model_name = dicom.get("ModelName", "Unknown")

            # Extract clinical information
            birads = dicom.get("0010,21D0", "Unknown")  # BI-RADS assessment

            print(f"File: {file}")
            print(f"  PatientID: {patient_id}")
            print(f"  PatientName: {patient_name}")
            print(f"  PatientBirthDate: {patient_birth_date}")
            print(f"  PatientSex: {patient_sex}")
            print(f"  StudyInstanceUID: {study_instance_uid}")
            print(f"  StudyDate: {study_date}")
            print(f"  SeriesInstanceUID: {series_instance_uid}")
            print(f"  SeriesNumber: {series_number}")
            print(f"  Manufacturer: {manufacturer}")
            print(f"  ModelName: {model_name}")
            print(f"  BI-RADS: {birads}")

            # Convert the DICOM pixel array to an image and save it
            image = dicom.pixel_array
            plt.imshow(image, cmap='gray')
            plt.title(f"PatientID: {dicom.PatientID}")
            plt.axis('off')
            output_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}.png")
            plt.savefig(output_path)
            plt.close()

            count += 1

if __name__ == "__main__":
    cmmd_path = "/data/bl70/validate/CMMD"
    cbis_ddsm_path = "/data/bl70/validate/CBIS-DDSM"
    output_dir = os.path.expanduser("~/CS5099-Code/src/data-processing/output")

    os.makedirs(output_dir, exist_ok=True)

    print("CMMD Patient IDs and Images:")
    load_and_extract_metadata(cmmd_path, output_dir, num_images=2)

    print("\nCBIS-DDSM Patient IDs and Images:")
    load_and_extract_metadata(cbis_ddsm_path, output_dir, num_images=2)
