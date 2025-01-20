import os
import pickle
from pathlib import Path
# import wget
from monai.networks.nets import resnet50
from .download_utils import bar_progress
from .load_model import LoadModel



def get_linear_classifier(weights_path=None, download_url="https://www.dropbox.com/s/77zg2av5c6edjfu/task3.pkl?dl=1"):
    if weights_path is None:
        weights_path = "/tmp/linear_model.pkl"
        wget.download(download_url, out=weights_path)

    return pickle.load(open(weights_path, "rb"))


def fmcib_model(eval_mode=False, ckpt_path=None, widen_factor=2, pretrained=False,
                bias_downsample=True, conv1_t_stride=2):
    trunk = resnet50(
        pretrained=pretrained,
        n_input_channels=1,
        widen_factor=widen_factor,
        conv1_t_stride=conv1_t_stride,
        feed_forward=False,
        bias_downsample=bias_downsample,
    )
    # weights_url = "https://zenodo.org/records/10528450/files/model_weights.torch?download=1"
    # current_path = Path(os.getcwd())
    # if not (current_path / "model_weights.torch").exists():
    #     wget.download(weights_url, bar=bar_progress)
    model = LoadModel(trunk=trunk, weights_path=ckpt_path, heads=[])

    if eval_mode:
        model.eval()

    return model