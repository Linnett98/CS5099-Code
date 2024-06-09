from tciaclient import TCIAClient
import time
import zipfile
from zipfile import ZipFile
import os
import logging
import argparse
import pandas as pd

def get_series_uids_for_collection_and_modality(collection, modality):
    # Load metadata for the collection
    collection_data = pd.read_csv(f"metadata/{collection}.csv")
    # Filter series UIDs by modality
    series_uids = collection_data[collection_data['Modality'] == modality]['SeriesInstanceUID'].tolist()
    return series_uids

parser = argparse.ArgumentParser()
parser.add_argument("--start", 
    required=False,
    default = "",
    help="specific dataset to start from")
parser.add_argument("dataset", 
    metavar="d",
    help="one of train, validate or test")
parser.add_argument("--out", 
    required=True,
    help="directory to save data in")
args = parser.parse_args()

output_dir = args.out
dataset = args.dataset
logging.basicConfig(filename=f'{dataset}.log', filemode="w", level=logging.DEBUG, format='%(levelname)s:%(message)s')

modalities = ["MR", "CT", "PT", "CR", "DX", "MG"]
limits = {key: 1e9 for key in modalities}

# Define the specific datasets to download
specific_datasets = ["CMMD", "CBIS-DDSM", "VICTRE"]

start_index = 0 
if args.start:
    print(f"Start at {args.start}")
    if args.start in specific_datasets:
        start_index = specific_datasets.index(args.start)
    else:
        raise ValueError(f"Start dataset {args.start} is not in the list of specific datasets: {specific_datasets}")

tcia_client = TCIAClient(baseUrl="https://services.cancerimagingarchive.net/services/v4", resource="TCIA")
start = time.time()
bytes_downloaded = 0

# Iterate over the specific datasets and download images
for collection_index in range(start_index, len(specific_datasets)):
    collection = specific_datasets[collection_index]
    print(f"Downloading collection: {collection}")
    logging.info(f"Downloading {collection}")
    
    for modality in modalities:
        series_uid_list = get_series_uids_for_collection_and_modality(collection, modality)
        for i, series_uid in enumerate(series_uid_list):
            print(f"Processing series {i} for collection {collection}", end="\r")
            try:
                zip_path = f"{output_dir}/temp-{collection}-{i}.zip"
                tcia_client.get_image(series_uid, output_dir, zip_path)
                with ZipFile(zip_path, "r") as zip_file:
                    zip_file.extractall(f"{output_dir}/{dataset}/{collection}")
                os.remove(zip_path)
            except zipfile.BadZipFile:
                logging.warning(f"Bad zip file: {collection} --- {series_uid}")
            except IOError:
                logging.warning(f"File not found, {collection} --- {series_uid}")

end = time.time()
print(f"Downloaded {bytes_downloaded // 1000000000}GB in {end-start} seconds")
