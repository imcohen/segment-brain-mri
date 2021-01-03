from skimage import io
import torch
from torch.utils.data import Dataset
import numpy as np


class LGGDataset(Dataset):
    """LGG dataset."""

    def __init__(self, df, transform=None):
        """
        Args:
            df (Dataframe): Pandas dataframe
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = io.imread(self.df.image_path[idx])[:, :, 1].astype(np.float32)/255
        mask = io.imread(self.df.mask_path[idx]).astype(np.float32)/255

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)

        if self.transform:
            image, mask = self.transform((image, mask))

        return torch.from_numpy(image), torch.from_numpy(mask)

