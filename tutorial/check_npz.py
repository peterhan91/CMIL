import os
import zipfile
import numpy as np
import logging
import json
from tqdm import tqdm

# Configure logging
logging.basicConfig(filename="bad_zip_files.log", level=logging.ERROR, format="%(asctime)s - %(message)s")

def check_npz_files(directory, output_json):
    """
    Check all .npz files in the specified directory for corruption and save bad files to JSON.

    Args:
        directory (str): Path to the directory containing .npz files.
        output_json (str): Path to save the bad files list as JSON.
    """
    bad_files = []

    # Traverse through the directory
    for root, _, files in os.walk(directory):
        for file in tqdm(files):
            if file.endswith(".npz"):
                file_path = os.path.join(root, file)
                try:
                    # Attempt to open the .npz file
                    with np.load(file_path, mmap_mode=None) as f:
                        _ = list(f.keys())  # Access file content to ensure it's readable
                except zipfile.BadZipFile:
                    # Log and collect bad file details
                    logging.error(f"BadZipFile: {file_path}")
                    print(f"BadZipFile found: {file_path}")
                    bad_files.append(file_path)
                except Exception as e:
                    # Log other exceptions
                    logging.error(f"Error with file {file_path}: {e}")
                    print(f"Error with file {file_path}: {e}")
                    bad_files.append(file_path)

    # Save the list of bad files to JSON
    with open(output_json, "w") as json_file:
        json.dump(bad_files, json_file, indent=4)

    # Summary
    print(f"Found {len(bad_files)} bad .npz files.")
    print(f"List of bad files saved to {output_json}")

if __name__ == "__main__":
    # Change these to your paths
    npz_directory = "/mnt/nas/CT/npz_npy"  # Directory containing .npz files
    output_json_path = "bad_files.json"        # Path to save the JSON file

    check_npz_files(npz_directory, output_json_path)
