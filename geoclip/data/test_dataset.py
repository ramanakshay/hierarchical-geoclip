import torch
import webdataset as wds
from PIL import Image
from torchvision import transforms
import pandas as pd


# csv_path = '/scratch/ar8692/image-geolocation/geoclip/data/datasets/metadata/MP16_Pro_filtered.csv'
# df = pd.read_csv(csv_path)
# print('No. of Images:', len(df))
# # No. of Images: 4122119

# image_transforms = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
#     transforms.RandomGrayscale(p=0.2),
#     transforms.PILToTensor(),
#     transforms.ConvertImageDtype(torch.float),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# ])

def preprocess(sample):
    image, json = sample
    lat, lon = float(json['LAT']), float(json['LON'])
    return image, torch.tensor((lat, lon))

url = 'datasets/mp-16-transformed/mp-16-transformed-{000000..000393}.tar'
dataset = wds.WebDataset(url, shardshuffle=True).shuffle(1000).decode("pil").to_tuple("jpg", "json").map(preprocess)
# dataloader = wds.WebLoader(dataset, batch_size=256)


for image, loc in iter(dataset):
    break
