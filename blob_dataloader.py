import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from PIL import Image

class BlobDataset(Dataset):

    def __init__(self, data_path, train, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.train = train
        self.labels = pd.read_csv(data_path + 'y_dataset.csv', index_col=0)

        self.train_lim = int(0.8 * self.labels.shape[0])

    def __len__(self):
        if self.train:
            return self.train_lim
        else:
            return self.labels.shape[0] - self.train_lim

    def __getitem__(self, index):
        if not self.train:
            index += self.train_lim
            
        image_path = self.data_path + str(index) + '.jpg'
        image = Image.open(image_path).convert('RGB')
        label = self.labels.loc[index]
        label[0] = int(label[0])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# Assume the data is stored in a list called 'data'


# Set batch size and whether or not to shuffle the data
batch_size = 32
shuffle = True




