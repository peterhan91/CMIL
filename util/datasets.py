from torchvision import transforms
from transforms import *
from dataset import Simple_Dataset

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    dataset = Simple_Dataset(csv_path=args.csv_path, 
                                 img_folder=args.lmdb_path,
                                 file_ext='npz',
                                 transforms=transform,
    )
    return dataset

def build_transform(is_train, args):
    if is_train:
        transform = transforms.Compose([
                    RandomResizedCrop3D(size=args.input_size),
                    RandomGamma3D(prob=0.5),
                    RandGaussianSmooth(sigma_range=(0.25, 1.5), prob=0.5),
                    ZScoreNormalizationPerSample()
                ])
        return transform
    
    t = []
    t.append(CenterCrop3D(size=args.input_size))
    t.append(ZScoreNormalizationPerSample())
    return transforms.Compose(t)