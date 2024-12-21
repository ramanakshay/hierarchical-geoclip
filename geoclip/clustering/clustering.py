from geoclip.model.GeoCLIP import *
import torch
import hdbscan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from collections import defaultdict
import json

# Initialize GeoCLIP model
model = GeoCLIP(from_pretrained=True)
model.to("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

location_encoder = model.location_encoder
print('Model loaded.')
locations = model.gps_gallery.to('cpu')
gps_data = pd.DataFrame(model.gps_gallery.cpu().numpy(), columns=["LAT", "LON"])

print(locations[:3])

nclusters = [50,10]
def get_labels(locations,sigma_index):
    if sigma_index == 2:
        return [str(loc) for loc in locations.tolist()]
    print(f"Clustering: |locations|={len(locations)} sigma_index={sigma_index}")
    with torch.no_grad():
        embeddings = location_encoder(locations.to(model.device), index=sigma_index).to('cpu')
    pca = PCA(n_components=10)
    pca_data = pca.fit_transform(embeddings)
    kmeans = KMeans(n_clusters=nclusters[sigma_index])
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

    
mydict = get_labels(locations,0)

print(mydict.keys())

with open("data.json", "w") as file:
    json.dump(mydict, file)













gps_data.to_csv("multiresolution_clustering.csv", index=False)



# plt.scatter(locations[:, 1], locations[:, 0], c=cluster_labels, cmap='tab20', s=10, alpha=0.7)
# plt.savefig('locations.png')

# #
# algorithm = Trainer(dataset, model, config)
# algorithm.run()


# # Save cluster results
# gps_data = pd.DataFrame(model.gps_gallery.cpu().numpy(), columns=["LAT", "LON"])
# gps_data["Cluster"] = cluster_labels

# gps_data.to_csv("clustered_gps_embeddings.csv", index=False)

# # Visualize embeddings using PCA
# def visualize_embeddings(embeddings, cluster_labels, title="Cluster Visualization"):
#     """Visualize high-dimensional embeddings in 2D using PCA."""
#     pca = PCA(n_components=2)
#     reduced_embeddings = pca.fit_transform(embeddings)
#     plt.figure(figsize=(10, 6))
#     scatter = plt.scatter(
#         reduced_embeddings[:, 0],
#         reduced_embeddings[:, 1],
#         c=cluster_labels,
#         cmap="viridis",
#         s=10,
#         alpha=0.7
#     )
#     plt.colorbar(scatter, label="Cluster ID")
#     plt.title(title)
#     plt.xlabel("PCA Component 1")
#     plt.ylabel("PCA Component 2")
#     plt.show()

# visualize_embeddings(embeddings, cluster_labels, title="HDBSCAN Clusters for GPS Embeddings")
