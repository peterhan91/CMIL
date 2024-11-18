# data_loader.py

import os
from monai.data import Dataset, DataLoader
from monai.transforms import (
    LoadImaged,
    AddChanneld,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandFlipd,
    RandRotate90d,
    ToTensord,
    Compose,
)
from sklearn.model_selection import train_test_split

def get_data_loaders(data_dir, batch_size=1, num_workers=4, test_size=0.2, random_state=42):
    """
    Creates training and validation DataLoaders for CT volume data.

    Args:
        data_dir (str): Directory containing CT volume files.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of worker processes for DataLoader.
        test_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Random seed for data splitting.

    Returns:
        Tuple[DataLoader, DataLoader]: DataLoaders for training and validation datasets.
    """
    # Prepare a list of data dictionaries
    data_dicts = []
    for fname in os.listdir(data_dir):
        if fname.endswith('.nii') or fname.endswith('.nii.gz'):
            image_path = os.path.join(data_dir, fname)
            # Extract label from filename or a separate file
            if 'positive' in fname.lower():
                label = 1
            else:
                label = 0
            data_dicts.append({'image': image_path, 'label': label})

    # Split data into training and validation sets
    train_files, val_files = train_test_split(
        data_dicts, test_size=test_size, random_state=random_state
    )

    # Define training transforms
    train_transforms = Compose([
        LoadImaged(keys=['image']),
        AddChanneld(keys=['image']),  # Add channel dimension
        Spacingd(keys=['image'], pixdim=(1.0, 1.0, 1.0), mode='bilinear'),
        Orientationd(keys=['image'], axcodes='RAS'),
        ScaleIntensityRanged(
            keys=['image'], a_min=-1000, a_max=1000,
            b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=['image'], source_key='image'),
        RandFlipd(keys=['image'], spatial_axis=[0, 1, 2], prob=0.5),
        RandRotate90d(keys=['image'], prob=0.5),
        ToTensord(keys=['image', 'label']),
    ])

    # Define validation transforms (without random augmentations)
    val_transforms = Compose([
        LoadImaged(keys=['image']),
        AddChanneld(keys=['image']),
        Spacingd(keys=['image'], pixdim=(1.0, 1.0, 1.0), mode='bilinear'),
        Orientationd(keys=['image'], axcodes='RAS'),
        ScaleIntensityRanged(
            keys=['image'], a_min=-1000, a_max=1000,
            b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=['image'], source_key='image'),
        ToTensord(keys=['image', 'label']),
    ])

    # Create datasets
    train_dataset = Dataset(data=train_files, transform=train_transforms)
    val_dataset = Dataset(data=val_files, transform=val_transforms)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader
