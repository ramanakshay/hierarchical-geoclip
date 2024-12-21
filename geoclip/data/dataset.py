import torch
import numpy as np
from torch.utils.data import IterableDataset, Dataset
import webdataset as wds
from torchvision import transforms
import pandas as pd

image_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# val_transforms = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.PILToTensor()
#     transforms.ConvertImageDtype(torch.float),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# ])

def preprocess(sample):
    image, json = sample
    lat, lon = float(json['LAT']), float(json['LON'])
    return image_transforms(image), torch.tensor((lat, lon))

class MP16Dataset(IterableDataset):
    def __init__(self, path):
        self.dataset = wds.WebDataset(path, shardshuffle=True).shuffle(1000).decode("pil").to_tuple("jpg", "json").map(preprocess)

    def __iter__(self):
        return iter(self.dataset)

class MP16EmbDataset(Dataset):
    def __init__(self, features_path, metadata_path):
        numpy_data = np.load(features_path)
        self.features = torch.from_numpy(numpy_data)
        self.metadata = pd.read_parquet(metadata_path)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, i):
        embedding = self.features[i]
        label = self.metadata.iloc[i]
        return embedding.float(), torch.tensor([label['LAT'],label['LON']]).float()


