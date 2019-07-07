import os
import pandas as pd
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import pil_loader
import pydicom
from mask_function import load_mask
from PIL import Image

class SIIMDataset(Dataset):
    """
    args:
    - phase    : phase of the dataset used (∈{"train", "test"})
    - transform:  
    """
    def __init__(self, phase, df, transform=None):
        self.phase = phase
        #self.root_dir = root_dir
        self.cols = 2 if phase == "train" else 1
        self.df = df #pd.read_csv(f"{root_dir}/{phase}.csv", usecols = [i for i in range(self.cols)])
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx][0]
        image = pydicom.read_file(img_path).pixel_array
        shape = np.array(image).shape
        image = Image.fromarray(image)
        label = None
        if self.phase == "train"
            label = self.df.loc[idx][1]
            label = load_mask(shape[0],shape[1],label)
        #image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)   # transform image to tensor
            if self.phase == "train":
                label = self.transform(label)
        #image = torch.stack(image)
        sample = {'image':image, 'label':label}
        return sample

def get_dataloaders(data, batch_size=8, study_level=False):
    '''
    Returns dataloader pipeline with data augmentation
    '''
    data_transforms = {
        'train': transforms.Compose([
                #transforms.Resize((224, 224)),
                #transforms.RandomHorizontalFlip(),
                #transforms.RandomRotation(10),
                transforms.ToTensor(),
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ]),
        'test': transforms.Compose([
            #transforms.Resize((224, 224)),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    phase=["train","test"]
    image_datasets = {x:SIIMDataset(x,data[x],transform=data_transforms[x]) for x in phase}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in phase}
    return dataloaders

if __name__=='main':
    pass
