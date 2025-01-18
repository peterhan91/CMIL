import pandas as pd
import numpy as np
import logging
import nibabel as nib
import lmdb
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class Simple_Dataset(Dataset):
    def __init__(self, csv_path, img_folder, file_ext='npz', 
                 transforms=None, sub_sample=False, task='ct-rate'):
        img_folder = Path(img_folder)
        files = list(img_folder.glob(f'*.{file_ext}'))
        # Build a mapping from VolumeName (with 'nii.gz' extension) to the actual file path
        self.file_dict = {
            f.name.replace(f'.{file_ext}', '.nii.gz'): f for f in files
        }

        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['VolumeName'].isin(self.file_dict.keys())].reset_index(drop=True)
        self.transforms = transforms
        self.img_folder = img_folder
        self.file_ext = file_ext
        
        if sub_sample:
            self.df = self.df.sample(n=10, random_state=42).reset_index(drop=True)
        
        if task == 'ct-rate':
            self.columes = [
                'Medical material', 'Arterial wall calcification', 'Cardiomegaly',
                'Pericardial effusion', 'Coronary artery wall calcification',
                'Hiatal hernia', 'Lymphadenopathy', 'Emphysema', 'Atelectasis',
                'Lung nodule', 'Lung opacity', 'Pulmonary fibrotic sequela',
                'Pleural effusion', 'Mosaic attenuation pattern', 'Peribronchial thickening',
                'Consolidation', 'Bronchiectasis', 'Interlobular septal thickening',
            ]
        elif task == 'inspect':
            self.columes = [
                '1_month_mortality', '6_month_mortality', '12_month_mortality',
                '1_month_readmission', '6_month_readmission', '12_month_readmission',
                '12_month_PH',
                'pe_acute', 'pe_subsegmentalonly', 'pe_positive'
            ]
        elif task == 'respect':
            self.columes = [
                'positive_exam_for_pe', 
                'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1',
                'leftsided_pe', 'rightsided_pe', 'central_pe', 
                'acute_and_chronic_pe', 'chronic_pe',
            ]
        elif task == 'nsclc_radiomics':
            self.columes = [
                'Overall.Stage.map'
            ]
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx): 
        row = self.df.iloc[idx]
        img_name = row['VolumeName']
        img_path = self.file_dict[img_name]

        if self.file_ext == 'npz':
            with np.load(img_path, mmap_mode=None) as f:
                img_data = f['img_data']
        elif self.file_ext == 'nii':
            img_data = nib.load(str(img_path)).get_fdata()
        else:
            raise NotImplementedError(f'File extension {self.file_ext} is not supported.')
        
        img_data = torch.from_numpy(img_data.astype(np.float32)).unsqueeze(0)
        if self.transforms:
            img_data = self.transforms(img_data)
        
        label = row[self.columes].values.astype(np.int32)
        
        return img_data, label


class LMDB_Dataset(Dataset):
    def __init__(self, csv_path, lmdb_path, image_key='VolumeName',
                 transforms=None, sub_sample=False):
        logging.debug(f'Loading image data from {lmdb_path}.')
        self.lmdb_path = lmdb_path
        self.df = pd.read_csv(csv_path)
        self.transforms = transforms
        self.image_key = image_key
        if sub_sample:
            self.df = self.df.sample(n=10, random_state=42).reset_index(drop=True)
        self.env = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path, readonly=True, lock=False, readahead=True, meminit=False
            )

        row = self.df.iloc[idx]
        key = row[self.image_key]
        with self.env.begin(write=False, buffers=True) as txn:
            value = txn.get(key.encode('ascii'))
            shape_value = txn.get(f"{key}_shape".encode('ascii'))
            img_shape = np.frombuffer(shape_value, dtype=np.int32)
            data = np.frombuffer(value, dtype=np.uint8).reshape(img_shape).copy()
            data = torch.from_numpy(data).unsqueeze(0).float().div(255)

        if self.transforms:
            data = self.transforms(data)
        
        label = []
        columes = ['Medical material', 'Arterial wall calcification', 'Cardiomegaly',
                   'Pericardial effusion', 'Coronary artery wall calcification',
                   'Hiatal hernia', 'Lymphadenopathy', 'Emphysema', 'Atelectasis',
                   'Lung nodule', 'Lung opacity','Pulmonary fibrotic sequela',
                   'Pleural effusion', 'Mosaic attenuation pattern', 'Peribronchial thickening',
                   'Consolidation', 'Bronchiectasis', 'Interlobular septal thickening']
        
        label = row[columes].values.astype(np.int32)
        
        return data, label
    
    def __getstate__(self):
        # Exclude the LMDB environment from being pickled
        state = self.__dict__.copy()
        state['env'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.env = None

    def __del__(self):
        # Close the LMDB environment when the dataset is destroyed
        if self.env is not None:
            self.env.close()

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


