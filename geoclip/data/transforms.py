from PIL import Image
import webdataset as wds
import pandas as pd
from itertools import islice
from torchvision import transforms
import os
import io
import torch
from tqdm import tqdm

# download dataset from hugging face
# snapshot_download(repo_id="Jia-py/MP16-Pro", repo_type='dataset', local_dir='/scratch/ar8692/image-geolocation/data/datasets')


dataset_path = '/scratch/ar8692/image-geolocation/geoclip/data/datasets/mp-16/mp-16-{000000..000393}.tar'
dataset = wds.WebDataset(dataset_path, shardshuffle=False)
iterator = iter(dataset)
print('Web Dataset loaded.')

image_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.1),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.ToPILImage(mode='RGB')
])


with wds.ShardWriter("/scratch/ar8692/image-geolocation/geoclip/data/datasets/mp-16-transformed/mp-16-transformed-%06d.tar", maxsize=1e9) as sink:
    for sample in tqdm(iterator, total=4150000):
        # check if image exists
        try:
            image = Image.open(io.BytesIO(sample['jpg']))
            image = image_transforms(image)
        except:
            continue
        output = io.BytesIO()
        image.save(output, format='JPEG')
        sample['jpg'] = output.getvalue()
        sink.write(sample)


