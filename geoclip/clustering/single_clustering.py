"""
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install Pillow transformers pandas numpy geopy
conda install -c rapidsai -c nvidia -c conda-forge \
    cuml=23.10 python=3.10 cudatoolkit=11.8
"""
from cuml.cluster import DBSCAN
from geoclip.model.GeoCLIP import *
import torch
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#from sklearn.decomposition import PCA

# Initialize GeoCLIP model
geoclip = GeoCLIP(from_pretrained=True)
geoclip.to("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

# Extract GPS embeddings
@torch.no_grad()
def extract_gps_embeddings(model):
    """Extract embeddings from the gps_gallery using the LocationEncoder."""
    gps_gallery = model.gps_gallery.to(model.device)  # Load GPS gallery
    embeddings = model.location_encoder(gps_gallery, index=0)  # Encode GPS coordinates
    #embeddings = F.normalize(embeddings, dim=1).cpu().numpy()  # Normalize and move to CPU
    return embeddings

gps_embeddings = extract_gps_embeddings(geoclip)

def cluster_embeddings_gpu(embeddings, eps=0.5, min_samples=10):
    """Cluster embeddings using GPU-accelerated DBSCAN."""
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    cluster_labels = db.fit_predict(embeddings)
    return cluster_labels

cluster_labels = cluster_embeddings_gpu(gps_embeddings, eps=0.1, min_samples=10)

# Save clusters with GPS coordinates
gps_data = pd.DataFrame(geoclip.gps_gallery.cpu().numpy(), columns=["LAT", "LON"])
gps_data["Cluster"] = cluster_labels  # Add cluster labels
output_file = "clustered_gps_data.csv"
gps_data.to_csv(output_file, index=False)  # Save to CSV

print(f"Clustered data saved to {output_file}")
