from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

