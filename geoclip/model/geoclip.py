import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .image_encoder import ImageEncoder
from .location_encoder import LocationEncoder
from .utils import load_gps_data, file_dir

from PIL import Image
from torchvision.transforms import ToPILImage

class GeoCLIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.image_encoder = ImageEncoder(config.use_embeddings, config.emb_dim)
        self.location_encoder = LocationEncoder()
        self.gps_gallery = load_gps_data(config.gallery_path)
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.to(self.device)

        self.weights_folder = config.weights_path
        if config.from_pretrained:
            self.load_weights()

    def load_weights(self):
        self.image_encoder.mlp.load_state_dict(torch.load(f"{self.weights_folder}image_encoder_mlp_weights_new.pth", weights_only=True))
        self.location_encoder.load_state_dict(torch.load(f"{self.weights_folder}location_encoder_weights_new.pth", weights_only=True))
        self.logit_scale = nn.Parameter(torch.load(f"{self.weights_folder}logit_scale_weights_new.pth", weights_only=True))

    def save_weights(self):
        torch.save(self.image_encoder.mlp.state_dict(), f'{self.weights_folder}image_encoder_mlp_weights_new.pth')
        torch.save(self.location_encoder.state_dict(), f'{self.weights_folder}location_encoder_weights_new.pth')
        torch.save(self.logit_scale, f'{self.weights_folder}logit_scale_weights_new.pth')

    def to(self, device):
        self.device = device
        self.image_encoder.to(device)
        self.location_encoder.to(device)
        self.logit_scale.data = self.logit_scale.data.to(device)
        self.gps_gallery = self.gps_gallery.to(device)
        return super().to(device)

    def forward(self, image, location, index=None):
        """ GeoCLIP's forward pass

        Args:
            image (torch.Tensor): Image tensor of shape (n, 3, 224, 224)
            location (torch.Tensor): GPS location tensor of shape (m, 2)

        Returns:
            logits_per_image (torch.Tensor): Logits per image of shape (n, m)
        """

        # Compute Features
        image_features = self.image_encoder(image)
        location_features = self.location_encoder(location, index)
        logit_scale = self.logit_scale.exp()

        # Normalize features
        image_features = F.normalize(image_features, dim=1)
        location_features = F.normalize(location_features, dim=1)

        # Cosine similarity (Image Features & Location Features)
        logits_per_image = logit_scale * (image_features @ location_features.t())

        return logits_per_image

    @torch.no_grad()
    def predict(self, image_path, top_k):
        """ Given an image, predict the top k GPS coordinates

        Args:
            image_path (str): Path to the image
            top_k (int): Number of top predictions to return

        Returns:
            top_pred_gps (torch.Tensor): Top k GPS coordinates of shape (k, 2)
            top_pred_prob (torch.Tensor): Top k GPS probabilities of shape (k,)
        """
        image = Image.open(image_path)
        image = self.image_encoder.preprocess_image(image)
        image = image.to(self.device)

        gps_gallery = self.gps_gallery.to(self.device)

        logits_per_image = self.forward(image, gps_gallery)
        probs_per_image = logits_per_image.softmax(dim=-1).cpu()

        # Get top k predictions
        top_pred = torch.topk(probs_per_image, top_k, dim=1)
        top_pred_gps = self.gps_gallery[top_pred.indices[0]]
        top_pred_prob = top_pred.values[0]

        return top_pred_gps, top_pred_prob