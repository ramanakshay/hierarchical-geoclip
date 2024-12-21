import json
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import ast
import random


def get_leaves(d):
    """Gets the leaf values of a nested dictionary."""
    leaves = []
    for key, value in d.items():
        if isinstance(value, dict):
            leaves.extend(get_leaves(value))
        else:
            leaves.append(value)
    return leaves

SEED = 4
with open("gps_gallery\data-20-50.json", "r") as file:
    clustered_dict = json.load(file)

# Initialize lists to store data
latitudes = []
longitudes = []
level_one_idxs = []
level_one_centers = []
level_two_idxs = []
level_two_centers = []

# Iterate through the nested dictionary
for level_one_idx, (level_one_center, level_two_dict) in enumerate(clustered_dict.items()):
    for level_two_idx, (level_two_center, element_list) in enumerate(level_two_dict.items()):
        for element in element_list:
            # Extract latitude and longitude from the string
            lat, lon = map(float, element.strip('[]').split(','))
            latitudes.append(lat)
            longitudes.append(lon)
            level_one_idxs.append(level_one_idx)
            level_one_centers.append(level_one_center)
            level_two_idxs.append(level_two_idx)
            level_two_centers.append(level_two_center)

# Create the DataFrame
df = pd.DataFrame({
    'LAT': latitudes,
    'LON': longitudes,
    'level_one_idx': level_one_idxs,
    'level_one_center': level_one_centers,
    'level_two_idx': level_two_idxs,
    'level_two_center': level_two_centers
})

FILE = "data.csv"

df.to_csv(FILE)

#### GEOPANDAS
import plotly.express as px


def visualize_level_two_clusters_with_background_zoomable(file_path, shapefile_path, mapbox_token, title="Zoomable Level-Two Clusters"):
    """
    Visualize level-two clusters with GeoPandas background on a zoomable Plotly map.

    Args:
        file_path (str): Path to the CSV file containing clustered data.
        shapefile_path (str): Path to the shapefile for country outlines.
        mapbox_token (str): Mapbox token for interactive maps.
        seed (int): Random seed for reproducibility.
        title (str): Title for the map.
    """
    # Load clustered data
    data = pd.read_csv(file_path)

    # Set random seed for reproducibility
    # Pick a random level-one cluster
    random.seed(SEED)
    random_level_one = random.choice(data["level_one_idx"].unique())
    print(f"Visualizing level-two clusters for level-one cluster: {random_level_one}")

    # Filter data for the selected level-one cluster
    subset = data[data["level_one_idx"] == random_level_one]

    # Parse centroids and create a combined cluster label for level-two clusters
    subset['level_two_cluster'] = subset[['level_one_idx', 'level_two_idx']].apply(tuple, axis=1)
    subset['level_two_center'] = subset['level_two_center'].apply(ast.literal_eval)

    # Load GeoPandas shapefile
    world = gpd.read_file(shapefile_path)

    # Convert GeoPandas data to GeoJSON
    geojson = world.to_json()

    # Initialize Plotly map
    fig = px.scatter_mapbox(
        subset,
        lat="LAT",
        lon="LON",
        color=subset['level_two_cluster'].astype(str),
        title=f"{title} (Level-One Cluster: {random_level_one})",
        hover_name="level_two_cluster",
        mapbox_style="carto-positron",
        zoom=3,
    )

    # Add GeoJSON layer for country outlines
    fig.update_layout(
        mapbox=dict(
            layers=[
                {
                    "source": geojson,
                    "type": "line",
                    "color": "black",
                    "opacity": 0.5,
                }
            ],
            center=dict(lat=subset["LAT"].mean(), lon=subset["LON"].mean()),
            zoom=6,
        ),
        mapbox_accesstoken=mapbox_token,
    )

    # Show the map
    fig.show()

# Example usage
mapbox_token = "pk.eyJ1Ijoic3Jpa2FudGhiYWxha3Jpc2huYSIsImEiOiJjbTRreWtiY3cwcTl5Mm5vcHBzODc0ajdzIn0.S3CoZLcyNzEwXmEdtQouYQ"  # Replace with your Mapbox token

shapefile_path = "CV/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp"  # Replace with your shapefile path



visualize_level_two_clusters_with_background_zoomable(FILE, shapefile_path, mapbox_token)


########


def visualize_level_two_clusters_for_one_level_one(file_path, title="Level-Two Clusters for a Random Level-One Cluster"):
    """
    Visualize level-two clusters for a random level-one cluster.

    Args:
        file_path (str): Path to the CSV file containing clustered data.
        title (str): Title for the plot.
    """
    # Load clustered data
    data = pd.read_csv(file_path)
    random.seed(SEED)
    # Pick a random level-one cluster
    random_level_one = random.choice(data["level_one_idx"].unique())
    print(f"Visualizing level-two clusters for level-one cluster: {random_level_one}")

    # Filter data for the selected level-one cluster
    subset = data[data["level_one_idx"] == random_level_one]

    # Create a combined cluster label for level-two clusters
    subset['level_two_cluster'] = subset[['level_one_idx', 'level_two_idx']].apply(tuple, axis=1)

    # Extract unique level-two clusters
    cluster_labels = subset['level_two_cluster'].values

    # Plot data points
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        subset["LON"], subset["LAT"], c=pd.factorize(cluster_labels)[0], cmap="tab20", s=10, alpha=0.7
    )
    plt.colorbar(scatter, label="Cluster ID")

    # Calculate centroids for level-two clusters
    subset['level_two_center'] = subset['level_two_center'].apply(ast.literal_eval)  # Convert string to list
    centroids = subset.groupby('level_two_cluster')['level_two_center'].first()  # Use first center for each cluster

    # Plot centroids
    for cluster, center in centroids.items():
        plt.scatter(center[1], center[0], c='black', s=25, marker='+')  # Lon is index 1, Lat is index 0

    # Add labels and title
    plt.title(f"{title} (Level-One Cluster: {random_level_one})")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.show()

# Example usage
visualize_level_two_clusters_for_one_level_one(FILE)



#############
#LEVEL-ONE-CLUSTERS
#############

def visualize_clusters_from_file(file_path, title="Top level clusters"):
    """
    Visualize clusters from a saved file containing latitude, longitude, and cluster labels.

    Args:
        file_path (str): Path to the CSV file containing clustered data.
        title (str): Title for the plot.
    """
    # Load clustered data
    data = pd.read_csv(file_path)
    
    # Extract clusters
    cluster_labels = data["level_one_idx"].values
    unique_clusters = set(cluster_labels)
    
    # Plot data
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        data["LON"], data["LAT"], c=cluster_labels, cmap="tab20", s=10, alpha=0.7
    )
    plt.colorbar(scatter, label="Cluster ID")
    
    # Calculate centroids for each cluster
    centroids = data.groupby("level_one_idx")[["LAT", "LON"]].mean()
    
    for cluster, row in centroids.iterrows():
        plt.scatter(row["LON"], row["LAT"], c='black', s=25, marker='+')
    
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.show()

# Example usage
visualize_clusters_from_file(FILE)

