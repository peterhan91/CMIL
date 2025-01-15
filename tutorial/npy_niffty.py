import os
import pydicom
import numpy as np
import nibabel as nib
import pandas as pd

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
import functools


def dicom_to_nifti(dicom_folder):
    """
    Convert a folder of DICOM slices (axial CT) to a .nii.gz volume.
    NOTE: Only suitable for straightforward axial acquisitions without complex orientation.
    """
    # Step 1: Gather all .dcm files
    dicom_files = [
        os.path.join(dicom_folder, f)
        for f in os.listdir(dicom_folder)
        if f.lower().endswith('.dcm')
    ]
    if not dicom_files:
        raise ValueError(f"No .dcm files found in folder: {dicom_folder}")

    # Step 2: Define a sorting key to order slices along z-axis
    def sorting_key(dicom_path):
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        # Try ImagePositionPatient first (most robust), else SliceLocation, else InstanceNumber
        if hasattr(ds, 'ImagePositionPatient'):
            # ImagePositionPatient is [x, y, z]; we use z to sort
            return float(ds.ImagePositionPatient[2])
        elif hasattr(ds, 'SliceLocation'):
            return float(ds.SliceLocation)
        else:
            return float(ds.InstanceNumber)

    dicom_files.sort(key=sorting_key)

    # Step 3: Read the first file to extract metadata
    ref_ds = pydicom.dcmread(dicom_files[0])
    # Basic error check if pixel data is missing
    if not hasattr(ref_ds, 'pixel_array'):
        raise ValueError(f"The first DICOM file has no pixel data: {dicom_files[0]}")

    # Step 4: Build the 3D numpy array (z, y, x)
    volume_slices = []
    for dcm_path in dicom_files:
        ds = pydicom.dcmread(dcm_path)
        pixel_array = ds.pixel_array  # 2D array for a single slice
        volume_slices.append(pixel_array)

    volume_3d = np.stack(volume_slices, axis=0)  # shape: (num_slices, height, width)

    # Step 5: Construct an approximate affine transform.
    px_spacing = getattr(ref_ds, "PixelSpacing", [1.0, 1.0])  # [row spacing, col spacing]
    slice_thickness = getattr(ref_ds, "SliceThickness", 1.0)

    affine = np.diag([
        float(px_spacing[1]),   # x voxel size
        float(px_spacing[0]),   # y voxel size
        float(slice_thickness), # z voxel size
        1.0
    ])

    # Step 6: Create a NIfTI image
    nifti_img = nib.Nifti1Image(volume_3d, affine)

    return nifti_img


def convert_and_save_nifti(file_name, source_root, save_dir):
    """
    Helper function for parallel processing. 
    - file_name: e.g. "ID_slice.nii.gz"
    - source_root: root directory of the DICOM folders
    - save_dir: where to save the resulting NIfTI file
    """
    # Prepare the input DICOM folder path (adjust logic to match your folder structure)
    parts = file_name.split('_')
    dicom_folder = os.path.join(source_root, parts[0], parts[1].split('.')[0])
    
    # Convert
    nii = dicom_to_nifti(dicom_folder)
    
    # Save
    save_path = os.path.join(save_dir, file_name)
    nib.save(nii, save_path)


if __name__ == '__main__':
    save_dir = '/mnt/nas/CT/RSNA_RESPECT/'
    source_root = '/mnt/nas/CT/rsna_pulmonary/train/'
    
    # DataFrame with metadata
    df = pd.read_csv('/home/than/DeepLearning/CMIL/csvs/RSNA_2020_metadata.csv')
    
    # Filter out volumes already processed
    processed = [os.path.basename(x) for x in glob(os.path.join(save_dir, '*.nii.gz'))]
    df = df[~df['VolumeName'].isin(processed)]
    
    os.makedirs(save_dir, exist_ok=True)

    # Convert column values to a list for easy mapping
    volume_names = df['VolumeName'].tolist()
    
    # Create a partial function that has source_root and save_dir pre-filled
    partial_convert_and_save_nifti = functools.partial(convert_and_save_nifti, source_root=source_root, save_dir=save_dir)

    # Create a pool with 8 worker processes
    with Pool(processes=8) as pool:
        for _ in tqdm(
            pool.imap_unordered(partial_convert_and_save_nifti, volume_names),
            total=len(volume_names),
            desc="Converting DICOM to NIfTI"
        ):
            pass

    print("All conversions completed!")

    print("All conversions completed!")
