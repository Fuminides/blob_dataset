import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from PIL import Image

class BlobDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.labels = pd.read_csv(data_path + 'y_dataset.csv', index_col=0)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = self.data_path + str(index) + '.jpg'
        image = Image.open(image_path).convert('RGB')
        label = self.labels.loc[index]
        if self.transform is not None:
            image = self.transform(image)

        return image, label


# Assume the data is stored in a list called 'data'
dataset = CustomDataset(data)

# Set batch size and whether or not to shuffle the data
batch_size = 32
shuffle = True

# Create the DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
Now you can iterate over the dataloader to get batches of data:

python
Copy code
for images, labels in dataloader:
    # do something with the images and labels




