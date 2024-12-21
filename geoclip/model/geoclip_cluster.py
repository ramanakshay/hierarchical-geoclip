import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .image_encoder import ImageEncoder
from .location_encoder import LocationEncoder
from .utils import load_gps_data, file_dir
from .gps_gallery.gallery import GPSGallery

from PIL import Image
from torchvision.transforms import ToPILImage

class GeoCLIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.image_encoder = ImageEncoder(config.use_embeddings, config.emb_dim)
        self.location_encoder = LocationEncoder()
        self.gps_gallery = GPSGallery(config.gallery_path)
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.to(self.device)

        if config.from_pretrained:
            self.weights_folder = config.weights_path
            self._load_weights()

    def _load_weights(self):
        self.image_encoder.mlp.load_state_dict(torch.load(f"{self.weights_folder}image_encoder_mlp_weights.pth", weights_only=True))
        self.location_encoder.load_state_dict(torch.load(f"{self.weights_folder}location_encoder_weights.pth", weights_only=True))
        self.logit_scale = nn.Parameter(torch.load(f"{self.weights_folder}logit_scale_weights.pth", weights_only=True))

    def to(self, device):
        self.device = device
        self.image_encoder.to(device)
        self.location_encoder.to(device)
        self.logit_scale.data = self.logit_scale.data.to(device)
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
    def predict(self, image):
        """ Given an image, predict the top k GPS coordinates

        Args:
            image_path (str): Path to the image

        Returns:
            top_pred_gps (torch.Tensor): Top k GPS coordinates of shape (k, 2)
            top_pred_prob (torch.Tensor): Top k GPS probabilities of shape (k,)
        """
        if type(image) == str:
            image = Image.open(image_path)
            image = self.image_encoder.preprocess_image(image)
            image = image.to(self.device)


        position = []
        for index in range(self.gps_gallery.levels+1):
            locations = self.gps_gallery.get_locations(position).to(self.device)
            logits = self.forward(image, locations, None)
            probs = logits.softmax(dim=-1)
            best = torch.argmax(probs, dim=-1).detach().cpu().numpy()
            position.append(int(best))


        best_gps = locations[best].detach().cpu().numpy()

        return best_gps