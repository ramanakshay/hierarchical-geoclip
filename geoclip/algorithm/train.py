import torch
from torch.utils.data import DataLoader
import webdataset as wds
from torch import nn, optim
from tqdm import tqdm
from datetime import datetime
import numpy as np
from geopy.distance import geodesic as GD

def distance_accuracy(targets, preds, dis=2500, gps_gallery=None):
    total = len(targets)
    correct = 0
    gd_avg = 0

    for i in range(total):
        gd = GD(gps_gallery[preds[i]], targets[i]).km
        if gd <= dis:
            gd_avg += gd
            correct += 1

    gd_avg /= correct
    return correct / total, gd_avg

def new_distance_accuracy(targets, preds, dis=2500, gps_gallery=None):
    total = len(targets)
    correct = 0
    gd_avg = 0

    for i in range(total):
        gd = GD(preds[i], targets[i]).km
        if gd <= dis:
            gd_avg += gd
            correct += 1

    gd_avg /= correct
    return correct / total, gd_avg


class Trainer(object):
    def __init__(self, dataset, model, config):
        self.config = config.algorithm
        self.model = model
        self.dataset = dataset

        # Construct dataloader
        if config.model.use_embeddings:
            self.train_dataloader = DataLoader(self.dataset['train'], batch_size=self.config.batch_size, drop_last=True, num_workers=4, shuffle=True)
            self.val_dataloader = DataLoader(self.dataset['val'],  batch_size=self.config.batch_size, drop_last=True, num_workers=4, shuffle=True)
        else:
            self.train_dataloader = wds.WebLoader(self.dataset['train'], batch_size=self.config.batch_size, drop_last=True)
            self.val_dataloader = wds.WebLoader(self.dataset['val'],  batch_size=self.config.batch_size, drop_last=True)

        # Model control.
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam([
            {'params': self.model.image_encoder.parameters(), 'lr': self.config.img_enc_learning_rate},
            {'params': self.model.location_encoder.parameters()},
            {'params': self.model.logit_scale}
        ], lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.87)
        self.device = self.model.device

    def train(self):
        self.model.train()
        bar = tqdm(self.train_dataloader)
        targets = torch.Tensor([i for i in range(self.config.batch_size)]).long().to(self.device)

        for (img, gps) in bar:
            img = img.to(self.device)
            gps = gps.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(img, gps)
            loss = self.loss_function(logits, targets)
            loss.backward()
            self.optimizer.step()
            bar.set_description("loss: {:.5f}".format(loss.item()))
        print("loss: {:.5f}".format(loss.item()))
        self.scheduler.step()

    def new_eval(self):
        self.model.eval()
        preds = []
        targets = []
        gps_gallery = self.model.gps_gallery
        with torch.no_grad():
            for imgs, labels in tqdm(self.val_dataloader, desc="Evaluating"):
                labels = labels.cpu().numpy()
                outs = []
                for i in range(imgs.shape[0]):
                    img = imgs[i].reshape(1,-1).to(self.device)
                    loc = self.model.predict(img)
                    outs.append(loc)

                preds.append(outs)
                targets.append(labels)

        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)

        distance_thresholds = [2500, 750, 200, 25, 1] # km
        accuracy_results = {}
        for dis in distance_thresholds:
            acc, avg_distance_error = new_distance_accuracy(targets, preds, dis, gps_gallery)
            print(f"Accuracy at {dis} km: {acc}, Average Distance Error: {avg_distance_error}")
            accuracy_results[f'acc_{dis}_km'] = acc

        return accuracy_results

    def eval(self):
        self.model.eval()
        preds = []
        targets = []
        gps_gallery = self.model.gps_gallery
        with torch.no_grad():
            for imgs, labels in tqdm(self.val_dataloader, desc="Evaluating"):
                labels = labels.cpu().numpy()
                imgs = imgs.to(self.device)
                # Get predictions (probabilities for each location based on similarity)
                logits_per_image = self.model(imgs, gps_gallery)
                probs = logits_per_image.softmax(dim=-1)

                # Predict gps location with the highest probability (index)
                outs = torch.argmax(probs, dim=-1).detach().cpu().numpy()

                preds.append(outs)
                targets.append(labels)

        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)

        distance_thresholds = [2500, 750, 200, 25, 1] # km
        accuracy_results = {}
        for dis in distance_thresholds:
            acc, avg_distance_error = distance_accuracy(targets, preds, dis, gps_gallery)
            print(f"Accuracy at {dis} km: {acc}, Average Distance Error: {avg_distance_error}")
            accuracy_results[f'acc_{dis}_km'] = acc

        return accuracy_results

    def run(self):
        # self.train()
        # self.eval()
        epochs = self.config.epochs
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            self.train()
            self.eval()
        self.model.save_weights()
        print("Done!")

