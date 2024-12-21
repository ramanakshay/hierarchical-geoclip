import json
import torch

def convert_to_tensor(locations):
    return torch.cat([torch.tensor(eval(location)).reshape(1, -1) for location in locations], dim=0)

class GPSGallery(object):
    def __init__(self, gallery_path):
        with open(gallery_path, 'r') as file:
            self.data = json.load(file)

    def get_value(self, dictionary, ind):
        return dictionary[list(dictionary.keys())[ind]]

    def get_locations(self, index=[]):
        if len(index) == 0:
            return convert_to_tensor(self.data.keys())
        elif len(index) == 1:
            return convert_to_tensor(self.get_value(self.data, index[0]).keys())
        else:
            return convert_to_tensor(self.get_value(self.get_value(self.data, index[0]), index[1]))

# gallery = GPSGallery('data.json')
#
# print(gallery.get_locations((1, 3)).size())