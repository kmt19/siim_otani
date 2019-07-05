from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch

class SIIMDataset(Dataset):
    """
    module to load dataset
    Some requirements:
    - path to the data should be described in csv file
    - path represents the image label like `/class1/img1.png`
    """
    def __init__(self, csv_file, root_dir, transform=None):
        self.img_paths = pd.read_csv(csv_file, sep = ',')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir,
                                self.img_paths.iloc[idx, 0])
        image = io.imread(img_path)
        match = re.findall('dataset/(.*)/', img_path)
        label = labels.index(match[0])

        image = Image.fromarray(image)
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)   # transform image to tensor

        sample = {'image':image, 'label':label}
        return sample
