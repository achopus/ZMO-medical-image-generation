import os
import json

import numpy as np
import nibabel as nib
from PIL import Image

import torch
from torch import Tensor
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader


def generate_images(json_path: str) -> None:
    """Generate train dataset from the given json file. Requires 3D volumes from "https://www.med.upenn.edu/sbia/brats2017.html".

    Args:
        json_path (str): `dataset.json` file in the downloaded files
    """

    dataset_json = json.load(open(json_path))

    # Merge train and test, since this split does not apply in our situation
    files_train = dataset_json["training"]
    files_valid = dataset_json["test"]
    files = [f["image"] for f in files_train]
    files.extend(f for f in files_valid)

    id_img = 1 # file id
    extend = 7 # number of pixels from the center of the volume, which should have their slice generated

    for idf, f in enumerate(files): # Each volume
        print(f"\r{idf + 1} / {len(files)}   ", end="")
        scan = nib.load(f)

        Z = scan.shape[-2]
        for r in range(-extend, extend): # Each layer around the center
            z = (Z // 2) + r
            image = scan.get_fdata()[:, :, z, 1]
            image = np.array(Image.fromarray(image).resize((256, 256), Image.BILINEAR))
            image_non_zero = image[image != image.min()].flatten() # Take into acount only brain pixels and discard the background
            image = (image - image_non_zero.mean()) / image_non_zero.var() # 0 mean, 1 variance
            image = 255 * (image - image.min()) / (image.max() - image.min()) # 0-1 normalization
            image_PIL = Image.fromarray(image.astype('uint8'))
            image_PIL.save(f"images/img_{str(id_img).zfill(4)}.png")
            id_img += 1

class DatasetImage(Dataset):
    """Generic dataset for image folder."""
    def __init__(self, folder: str, transform: v2.Transform = None) -> None:
        self.images = [os.path.join(folder, f) for f in os.listdir(folder)]
        self.N = len(self.images)
        self.T = transform
    
    def __getitem__(self, index) -> Tensor:
        image = Image.open(self.images[index])
        if self.T:
            return self.T(image)
        else:
            return torch.from_numpy(np.array(image))
    
    def __len__(self) -> int:
        return self.N

def get_dataloader_image(folder: str, batch_size: int = 64, image_size: int = 64) -> DataLoader:
    """Wrapper around the `DatasetImage`, which automaticly generates given `DataLoader`.

    Args:
        folder (str): Folder, where images are saved.
        batch_size (int, optional): Batch size. Defaults to `64`.
        image_size (int, optional): Size, to which the image will be interpolated. Defaults to `64`.

    Returns:
        DataLoader: _description_
    """
    transform = v2.Compose([
        v2.ToImage(),
        v2.Grayscale(),
        v2.Resize(image_size, antialias=True),
        v2.RandomVerticalFlip(),
        v2.RandomAffine(degrees=3, translate=(0.03, 0.03), scale=(0.95, 1.05)),
        v2.RandomPhotometricDistort(brightness=(0.95, 1.05), contrast=(0.95, 1.05), p=0.9),
        v2.ToDtype(torch.float32, scale=True)
    ])
    dataset = DatasetImage(folder, transform)
    return DataLoader(dataset, batch_size)

if __name__ == "__main__":
    generate_images("dataset.json")