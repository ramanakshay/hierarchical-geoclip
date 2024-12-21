from PIL import Image
import webdataset as wds
import pandas as pd
from itertools import islice
from torchvision import transforms
import os
import io
from huggingface_hub import snapshot_download

# download dataset from hugging face
snapshot_download(repo_id="Jia-py/MP16-Pro", repo_type='dataset', local_dir='/scratch/ar8692/image-geolocation/data/datasets')

csv_path = '/scratch/ar8692/image-geolocation/data/datasets/metadata/MP16_Pro_places365.csv'
df = pd.read_csv(csv_path)
df['IMG_ID'] = df['IMG_ID'].replace({'/': '_', '.jpg': ''}, regex=True)
df.sort_values(by='IMG_ID', inplace=True)
df.to_csv(csv_path, index=False)
print('Dataframe loaded.')

dataset_path = '/scratch/ar8692/image-geolocation/data/datasets/mp-16-images.tar'
dataset = wds.WebDataset(dataset_path, shardshuffle=False)
iterator = iter(dataset)
print('Web Dataset loaded.')

with wds.ShardWriter("datasets/mp-16/mp-16-%06d.tar", maxsize=1e9) as sink:
    for sample in iterator:
        # update key
        sample['__key__'] = os.path.basename(sample['__key__'])
        if len(sample['__key__']) == 0: continue

        # check if image exists
        try:
            image = Image.open(io.BytesIO(sample['jpg']))
        except:
            continue

        # search for metadata in csv
        result = df.iloc[df['IMG_ID'].searchsorted(sample['__key__'])]
        result = result.to_json().encode('utf-8')
        sample['json'] = result

        # write to shard
        sink.write(sample)


