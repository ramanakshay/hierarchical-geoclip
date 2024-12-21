from model import GeoCLIP
from data.dataset import MP16Dataset, MP16EmbDataset
from algorithm.train import Trainer
import webdataset as wds
import torch

from torch.utils.data import random_split

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config : DictConfig) -> None:
    # DATASET
    full_dataset = MP16EmbDataset(config.data.features_path, config.data.metadata_path)
    orig_dataset = wds.WebDataset(config.data.train_path, shardshuffle=True).shuffle(1000).decode("pil").to_tuple("jpg", "json")

    image, json  = next(iter(orig_dataset))
    print(json['IMG_ID'])
    df = full_dataset.metadata
    i = df.index[df['IMG_ID'] == json['IMG_ID']].tolist()[0]
    print(i)
    print(json['LAT'], json['LON'])
    embedding = full_dataset[i][0]
    label = full_dataset[i][1]
    print(label)

    model = GeoCLIP(config.model)
    x = model.image_encoder.preprocess_image(image)
    x = model.image_encoder.CLIP.get_image_features(x)
    x = x.squeeze()

    # x = full_dataset[i+2][0]
    # print(full_dataset[i+2][1])
    sim = torch.nn.functional.cosine_similarity(x, embedding, dim=0)
    print(sim)


    #
    # dataset = {
    #     'train': train_dataset,
    #     'val': val_dataset
    # }
    #
    # # # MODEL
    # model = GeoCLIP(config.model)
    # print('Model loaded.')
    # #
    # algorithm = Trainer(dataset, model, config)
    # algorithm.run()

if __name__ == "__main__":
    main()
