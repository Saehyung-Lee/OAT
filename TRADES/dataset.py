from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class TinyImageNet(Dataset):
    def __init__(self, img, transform=None):
        self.img = img.transpose((0, 2, 3, 1))
        self.transform = transform
        self.len = self.img.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        img = (Image.fromarray((self.img[index]).astype(np.uint8)))

        if self.transform is not None:
            img = self.transform(img)
        return img
