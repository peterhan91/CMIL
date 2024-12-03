import pandas as pd
import numpy as np
import logging
import nibabel as nib
import SimpleITK as sitk
import lmdb
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class NIB_Dataset(Dataset):
    def __init__(self, csv_path, target_shape=(384, 512, 512), transforms=None, sub_sample=False):
        self.df = pd.read_csv(csv_path)
        self.target_shape = target_shape
        self.transforms = transforms
        if sub_sample:
            self.df = self.df.sample(n=10, random_state=42).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx): 
        row = self.df.iloc[idx]
        img_path = row['Path']
        img_data = sitk.ReadImage(img_path)
        img_data = sitk.GetArrayFromImage(img_data)
        # Apply rescale slope and intercept
        img_data = float(row["RescaleSlope"]) * img_data + float(row["RescaleIntercept"])
        hu_min, hu_max = -1000, 1000
        img_data = np.clip(img_data, hu_min, hu_max)
        img_data = ((img_data + 1000) / 2000).astype(np.float16)  # Normalize to [0, 1]
        img_data = img_data.transpose(2, 1, 0)
        
        tensor = torch.from_numpy(img_data)
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        img_data = F.interpolate(tensor, size=self.target_shape, 
                        mode='trilinear', 
                        align_corners=False)
        img_data = img_data[0]

        if self.transforms:
            img_data = self.transforms(img_data)
        
        label = []
        columes = ['Medical material','Arterial wall calcification','Cardiomegaly',
                   'Pericardial effusion','Coronary artery wall calcification',
                   'Hiatal hernia','Lymphadenopathy','Emphysema','Atelectasis',
                   'Lung nodule','Lung opacity','Pulmonary fibrotic sequela',
                   'Pleural effusion','Mosaic attenuation pattern','Peribronchial thickening',
                   'Consolidation','Bronchiectasis','Interlobular septal thickening']
        label = row[columes].values.astype(np.int32)
        
        return img_data, label


class LMDB_Dataset(Dataset):
    def __init__(self, csv_path, lmdb_path, image_key='VolumeName',
                 transforms=None, target_shape=None, sub_sample=False):
        logging.debug(f'Loading image data from {lmdb_path}.')
        self.lmdb_path = lmdb_path
        self.df = pd.read_csv(csv_path)
        self.transforms = transforms
        self.image_key = image_key
        self.target_shape = target_shape
        if sub_sample:
            self.df = self.df.sample(n=10, random_state=42).reset_index(drop=True)
        self.env = lmdb.open(
                            lmdb_path,
                            max_readers=32,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False,
                        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', lmdb_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        key = row[self.image_key]
        with self.env.begin(write=False) as txn:
            value = txn.get(key.encode('ascii'))
            shape_value = txn.get(f"{key}_shape".encode('ascii'))
            img_shape = self.target_shape if self.target_shape else np.frombuffer(shape_value, dtype=np.int32)
            data = np.frombuffer(value, dtype=np.uint8).reshape(img_shape).copy()
            data = torch.from_numpy(data).unsqueeze(0).float().div(255)
        
        label = []
        columes = ['Medical material', 'Arterial wall calcification', 'Cardiomegaly',
                   'Pericardial effusion', 'Coronary artery wall calcification',
                   'Hiatal hernia', 'Lymphadenopathy', 'Emphysema', 'Atelectasis',
                   'Lung nodule', 'Lung opacity','Pulmonary fibrotic sequela',
                   'Pleural effusion', 'Mosaic attenuation pattern', 'Peribronchial thickening',
                   'Consolidation', 'Bronchiectasis', 'Interlobular septal thickening']
        
        label = row[columes].values.astype(np.int32)
        
        if self.transforms:
            data = self.transforms(data)
        
        return data, label


if __name__ == '__main__':
    csv_path = '/home/than/DeepLearning/CMIL/csvs/ct_rate_train_512.csv'
    lmdb_path = '/mnt/nas/Datasets/than/CT/LMDB/ct_rate_train_512.lmdb'
        
    # Create the dataset using LMDB_Dataset
    dataset_train = LMDB_Dataset(
        csv_path=csv_path,
        lmdb_path=lmdb_path,
        transforms=None,
    )

    print(f"Dataset loaded with {len(dataset_train)} samples.")
    
    data_loader_train = DataLoader(dataset_train, batch_size=32, num_workers=10,
                                    pin_memory=True, drop_last=True)

    for i, (data, label) in enumerate(data_loader_train):
        print(data.shape, label.shape)
        if i == 10:
            break


