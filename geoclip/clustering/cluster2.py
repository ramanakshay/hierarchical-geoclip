import torch
from model import GeoCLIP
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from collections import defaultdict
import json

from torch.utils.data import random_split

from data.dataset import MP16Dataset, MP16EmbDataset
from algorithm.train import Trainer


import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config : DictConfig) -> None:
    if config.model.use_embeddings:
        full_dataset = MP16EmbDataset(config.data.features_path, config.data.metadata_path)
        train_dataset, val_dataset = random_split(full_dataset, [0.999, 0.001])
    else:
        train_dataset = MP16Dataset(config.data.train_path)
        val_dataset = MP16Dataset(config.data.val_path)
    dataset = {
        'train': train_dataset,
        'val': val_dataset
    }
    print('Dataset loaded.')
    # MODEL
    model = GeoCLIP(config.model)
    location_encoder = model.location_encoder
    print('Model loaded.')

    # algorithm = Trainer(dataset, model, config)
    # algorithm.train()
    # algorithm.eval()
    # print('Model finetuned.')

    def get_labels(locations, sigma_index):
        if sigma_index == level:
            return [str(loc) for loc in locations.tolist()]

        print(f"Clustering: |locations|={len(locations)} sigma_index={sigma_index}")
        with torch.no_grad():
            embeddings = location_encoder(locations.to(model.device), index=sigma_index).to('cpu')
        pca = PCA(n_components=ncomponents[sigma_index])
        pca_data = pca.fit_transform(embeddings)
        kmeans = KMeans(n_clusters=min(nclusters[sigma_index], pca_data.shape[0]))
        kmeans.fit(pca_data)
        labels = kmeans.labels_

        cluster_dict = defaultdict(list)

        for i, cluster_id in enumerate(labels):
            cluster_dict[cluster_id].append(locations[i].reshape(1, -1))

        temp =  defaultdict(list)
        for cluster_id, locs in cluster_dict.items():
            locs = torch.cat(locs, dim=0)
            temp[str(torch.mean(locs, dim=0).tolist())] = locs
        cluster_dict = temp

        for centroid in cluster_dict.keys():
            cluster_dict[centroid] = get_labels(cluster_dict[centroid], sigma_index+1)

        return cluster_dict
    #
    # # level one trees
    for size in [150, 180, 200]:
        print(f'Size: {size}')
        locations = model.gps_gallery.to('cpu')
        nclusters = [size]
        ncomponents = [42, 20]
        level = 1
        mydict = get_labels(locations, 0)

        with open(f"new_data-{size}.json", "w") as file:
            json.dump(mydict, file)
    #
    # # level two trees
    # sizes = [[5,10], [10,10], [20,20], [20,50], [20,100], [100,10], [100,20], [200,10]]
    # for size in sizes:
    #     print(f'Size: {size}')
    #     locations = model.gps_gallery.to('cpu')
    #     nclusters = size
    #     level = 2
    #     mydict = get_labels(locations, 0)
    #
    #     with open(f"data-{size[0]}-{size[1]}.json", "w") as file:
    #         json.dump(mydict, file)
    # size = [150]
    # print(f'Size: {size}')
    # locations = model.gps_gallery.to('cpu')
    # nclusters = size
    # ncomponents = [42, 20]
    # level = 2
    # mydict = get_labels(locations, 0)

    # with open(f"new_data-{size[0]}-{size[1]}.json", "w") as file:
    #     json.dump(mydict, file)
    # #
    # algorithm = Trainer(dataset, model, config)
    # algorithm.run()

if __name__ == "__main__":
    main()

