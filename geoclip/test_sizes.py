from model.geoclip_cluster import GeoCLIP
# from model import GeoCLIP
from data.dataset import MP16Dataset, MP16EmbDataset
from algorithm.train import Trainer

from torch.utils.data import random_split

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config : DictConfig) -> None:
    # DATASET
    if config.model.use_embeddings:
        full_dataset = MP16EmbDataset(config.data.features_path, config.data.metadata_path)
        train_dataset, val_dataset = random_split(full_dataset, [0.99, 0.01])
    else:
        train_dataset = MP16Dataset(config.data.train_path)
        val_dataset = MP16Dataset(config.data.val_path)

    dataset = {
        'train': train_dataset,
        'val': val_dataset
    }
    print('Dataset Loaded')
    base_name = config.model.gallery_path
    # # MODEL
    sizes = ['100', '200', '300', '400', '500', '600', '700', '800', '900', '1000']
    sizes = sizes + ['5-10','10-10', '20-20', '20-50', '20-100', '100-10', '100-30', '200-10']
    for size in sizes:
        config.model.gallery_path = base_name + 'data-'+size+ '.json'
        print('Testing Size:', config.model.gallery_path)
        model = GeoCLIP(config.model)
        print('Model loaded.')
        #
        algorithm = Trainer(dataset, model, config)
        algorithm.run()

if __name__ == "__main__":
    main()
