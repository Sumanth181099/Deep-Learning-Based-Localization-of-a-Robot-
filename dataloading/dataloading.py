#!/usr/bin/python
import torch
from torch.utils.data import Dataset
import pandas as pd
from skimage import io
# import torchvision.transforms as transforms
import os


class ImageDataset(Dataset):

    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(csv_file, header=None)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir,
                                self.annotations.iloc[index, 0])
        # image = io.imread(self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_xlabel = torch.tensor(float(self.annotations.iloc[index, 1]))
        y_ylabel = torch.tensor(float(self.annotations.iloc[index, 2]))
        y_olabel = torch.tensor(float(self.annotations.iloc[index, 3]))
        target = torch.tensor([y_xlabel, y_ylabel, y_olabel],
                              dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return(image, target)


""" def main():
    i = ImageDataset(root_dir='resized_img_frames/resized_img_frames',
                     csv_file="training_data.csv",
                     transform=transforms.ToTensor())
    x = i.getitem(0)
    print(x.size())


if __name__ == "__main__":
    main()
 """
