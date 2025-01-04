import os
import nibabel as nib
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from glob import glob
from multiprocessing import Pool
from functools import partial


def read_nii_data(file_path):
    """
    Read NIfTI file data.

    Args:
    file_path (str): Path to the NIfTI file.

    Returns:
    np.ndarray: NIfTI file data.
    """
    try:
        nii_img = nib.load(file_path)
        nii_data = nii_img.get_fdata()
        return nii_data
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def resize_array(array, new_shape):
    return F.interpolate(array, size=new_shape,
                         mode='trilinear',
                         align_corners=False).cpu().numpy()


def process_image(row_dict, save_path, new_shape=(384, 512, 512)):
    row = pd.Series(row_dict)
    img_name = row['VolumeName']
    try:
        file_path = row['Path']
    except KeyError:
        file_path = os.path.join('/mnt/nas/CT/ct_rate_volumes/dataset/valid', row['VolumeName'].split("_")[0]+"_"+row['VolumeName'].split("_")[1], row['VolumeName'].split("_")[0]+"_"+row['VolumeName'].split("_")[1]+"_"+row['VolumeName'].split("_")[2], row['VolumeName'])

    slope = float(row["RescaleSlope"])
    intercept = float(row["RescaleIntercept"])

    # Read and process the image
    img_data = read_nii_data(file_path)
    if img_data is None:
        print(f"Read {file_path} unsuccessful. Skipping.")
        return None

    # Apply rescale slope and intercept
    img_data = slope * img_data + intercept
    hu_min, hu_max = -1000, 1000
    img_data = np.clip(img_data, hu_min, hu_max)
    img_data = ((img_data + 1000) / 2000).astype(np.float16)  # Normalize to [0, 1]
    img_data = img_data.transpose(2, 1, 0)

    # Convert to tensor and resize
    tensor = torch.tensor(img_data)
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    img_data = resize_array(tensor, new_shape)
    img_data = img_data[0, 0]  # Remove extra dimensions
    # img_data = (img_data * 255).astype(np.uint8)  # Convert to uint8 with range [0, 255]

    # Save as nii.gz file
    img_path = os.path.join(save_path, f"{img_name.split('.')[0]}.npz")
    # nib.save(nib.Nifti1Image(img_data, np.eye(4)), img_path)
    np.savez_compressed(img_path, img_data = img_data)

    return 1  # Return a value to indicate success


def convert_to_nii(annotations_csv, save_path, new_shape=(384, 512, 512)):
    # Load annotations CSV
    df = pd.read_csv(annotations_csv)
    processed = [os.path.basename(x).replace('npz', 'nii.gz') for x in list(glob(os.path.join(save_path, '*.npz')))]
    df = df[~df['VolumeName'].isin(processed)]

    # Prepare arguments
    rows = [row.to_dict() for _, row in df.iterrows()]

    with Pool(8) as pool:
        process_image_partial = partial(process_image, save_path=save_path, new_shape=new_shape)
        for _ in tqdm(pool.imap_unordered(process_image_partial, rows), total=len(rows)):
            pass

    print("NIfTI dataset creation complete.")


if __name__ == '__main__':
    # Example usage
    annotations_file = '../csvs/dataset_metadata_validation_metadata.csv'  # CSV file with image-text annotations
    os.makedirs('/mnt/nas/CT/npz_npy_valid', exist_ok=True)
    output_path = '/mnt/nas/CT/npz_npy_valid'  # Output path for the NIfTI dataset
    convert_to_nii(annotations_file, output_path)
