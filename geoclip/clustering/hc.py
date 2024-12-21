import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from geopy.distance import great_circle

file_path = 'CV/coordinates_100K.csv'  # Update this with the path to your CSV file
data = pd.read_csv(file_path)

# DBSCAN with geospatial clustering
def geospatial_dbscan(data, eps, min_samples):
    # Convert GPS to radians
    coords = np.radians(data[['LAT', 'LON']])
    if coords.shape[0] == 0:  # Check for empty dataset
        data['cluster'] = -1  # Mark all as noise
        return data
    # DBSCAN clustering
    dbscan = DBSCAN(eps=eps / 6371.0, min_samples=min_samples, metric='haversine').fit(coords)
    data.loc[:, 'cluster'] = dbscan.labels_  # Use .loc to assign in-place
    return data

# Level 1 clustering (large-scale regions, e.g., countries)
level1_eps = 1000  # Approx distance in km for level 1
level1_min_samples = 10
data_level1 = data.copy()
data_level1 = geospatial_dbscan(data_level1, level1_eps, level1_min_samples)

# Level 2 clustering (medium-scale regions, e.g., states)
level2_eps = 200  # Approx distance in km for level 2
level2_min_samples = 5
data_level2 = data_level1.copy()
data_level2['level2_cluster'] = -1
for cluster in data_level1['cluster'].unique():
    if cluster == -1:  # Skip noise points
        continue
    sub_data = data_level1[data_level1['cluster'] == cluster]
    clustered = geospatial_dbscan(sub_data.copy(), level2_eps, level2_min_samples)
    data_level2.loc[sub_data.index, 'level2_cluster'] = clustered['cluster']

# Level 3 clustering (small-scale regions, e.g., cities)
level3_eps = 20  # Approx distance in km for level 3
level3_min_samples = 2
data_level3 = data_level2.copy()
data_level3['level3_cluster'] = -1
for cluster in data_level2['level2_cluster'].unique():
    if cluster == -1:  # Skip noise points
        continue
    sub_data = data_level2[data_level2['level2_cluster'] == cluster]
    clustered = geospatial_dbscan(sub_data.copy(), level3_eps, level3_min_samples)
    data_level3.loc[sub_data.index, 'level3_cluster'] = clustered['cluster']

# Visualization
def plot_clusters(data, level_col, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(data['LON'], data['LAT'], c=data[level_col], cmap='viridis', s=5)
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.colorbar(label='Cluster ID')
    plt.show()

# Plot results
plot_clusters(data_level1, 'cluster', "Level 1 Clusters")
plot_clusters(data_level2, 'level2_cluster', "Level 2 Clusters")
plot_clusters(data_level3, 'level3_cluster', "Level 3 Clusters")
