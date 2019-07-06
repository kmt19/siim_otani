from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch
import pandas as pd
from skimage import io

class SIIMDataset(Dataset):
    """
    args:
    - phase    : phase of the dataset used (∈{"train", "test"})
    - transform:  
    """
    def __init__(self, root_dir, phase, transform=None):
        self.phase = phase
        self.root_dir = root_dir
        self.cols = 2 if phase == "train" else 1
        self.table = pd.read_csv(f"{root_dir}/{phase}.csv", usecols = [i for i in range(self.cols)])
        self.transform = transform

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir + f"/dicom_images_{self.phase}",
                                self.table["ImageId"][idx])
        image = io.imread(img_path)
        label = self.table["EncodedPixels"][idx] if self.phase == "train" else None

        image = Image.fromarray(image)
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)   # transform image to tensor

        sample = {'image':image, 'label':label}
        return sample
