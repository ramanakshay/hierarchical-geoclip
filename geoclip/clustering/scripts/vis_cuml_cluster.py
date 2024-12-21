import pandas as pd
import matplotlib.pyplot as plt

def visualize_clusters_from_file(file_path, title="Cluster Visualization"):
    """
    Visualize clusters from a saved file containing latitude, longitude, and cluster labels.

    Args:
        file_path (str): Path to the CSV file containing clustered data.
        title (str): Title for the plot.
    """
    # Load clustered data
    data = pd.read_csv(file_path)
    
    # Extract clusters
    cluster_labels = data["Cluster"].values
    unique_clusters = set(cluster_labels)
    
    # Plot data
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        data["LON"], data["LAT"], c=cluster_labels, cmap="tab20", s=10, alpha=0.7
    )
    plt.colorbar(scatter, label="Cluster ID")
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.show()

# Example usage
visualize_clusters_from_file("clustered_gps_data.csv")
