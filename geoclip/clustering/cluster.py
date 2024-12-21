import torch
import numpy as np
from model import GeoCLIP
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from torch.utils.data import random_split

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config : DictConfig) -> None:
    # MODEL
    model = GeoCLIP(config.model)
    location_encoder = model.location_encoder
    print('Model loaded.')

    locations = model.gps_gallery.to('cpu')
    with torch.no_grad():
        embeddings = location_encoder(model.gps_gallery, index=0).to('cpu')

    print(locations.size())
    print(embeddings.size())

    # pca = PCA().fit(embeddings)
    # plt.plot(np.cumsum(pca.explained_variance_ratio_)[:50])
    # plt.xlabel('number of components')
    # plt.ylabel('cumulative explained variance')
    # plt.savefig('variance.png')
    pca = PCA(n_components=42)
    pca_data = pca.fit_transform(embeddings)
    #
    # K-means
    kmeans = KMeans(n_clusters=200)
    kmeans.fit(pca_data)
    cluster_labels = kmeans.labels_

    # plt.scatter(locations[:, 1], locations[:, 0], c=cluster_labels, cmap='tab20', s=10, alpha=0.7)
    # plt.savefig('locations.png')

    # #
    # algorithm = Trainer(dataset, model, config)
    # algorithm.run()
    #
    # k_values = range(10, 400, 10)
    # inertias = []
    #
    # # Iterate through different K values
    # for k in k_values:
    #     print(f'Calculating for {k} clusters.')
    #     kmeans = KMeans(n_clusters=k)
    #     kmeans.fit(pca_data)
    #     cluster_labels = kmeans.labels_
    #     inertias.append(kmeans.inertia_)
    #
    #
    #
    # plt.plot(k_values, inertias)
    # plt.xlabel("Number of Clusters (K)")
    # plt.ylabel("Inertia")
    # plt.title("Inertia vs. Number of Clusters")
    #
    # # plt.scatter(locations[:, 1], locations[:, 0], c=cluster_labels, cmap='tab20', s=10, alpha=0.7)
    # plt.savefig('inertia_scores_bad.png')
    #
    target = 0
    indices = []
    n_clusters = 200
    for ind, label in enumerate(cluster_labels):
        if label == target:
            indices.append(ind)

    indices = torch.tensor(indices)
    locations = torch.index_select(locations, dim=0, index=indices)
    print(locations.size(), indices.size())

    with torch.no_grad():
        embeddings = location_encoder(locations.to(model.device), index=1).to('cpu')

    print(locations.size())
    print(embeddings.size())

    # pca = PCA(n_components=42)
    # pca_data = pca.fit_transform(embeddings)
    pca = PCA().fit(embeddings)
    plt.plot(np.cumsum(pca.explained_variance_ratio_)[:50])
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.savefig('variance_two.png')
    exit(0)

    k_values = range(10, 200, 5)
    inertias = []

    # Iterate through different K values
    for k in k_values:
        print(f'Calculating for {k} clusters.')
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(pca_data)
        cluster_labels = kmeans.labels_
        inertias.append(kmeans.inertia_)



    plt.plot(k_values, inertias)
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Inertia vs. Number of Clusters")

    # plt.scatter(locations[:, 1], locations[:, 0], c=cluster_labels, cmap='tab20', s=10, alpha=0.7)
    plt.savefig('inertia_scores_two.png')
    #
    # # K-means
    # kmeans = KMeans(n_clusters=50)
    # kmeans.fit(pca_data)
    # cluster_labels = kmeans.labels_
    #
    # plt.scatter(locations[:, 1], locations[:, 0], c=cluster_labels, cmap='tab20', s=10, alpha=0.7)
    # plt.savefig('locations.png')

    # #
    # algorithm = Trainer(dataset, model, config)
    # algorithm.run()

if __name__ == "__main__":
    main()
