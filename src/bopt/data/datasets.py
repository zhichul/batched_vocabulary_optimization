from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import os
import pickle


class LazyDataset(Dataset):

    def __init__(self, root):
        self.root = root

    def __len__(self):
        return len(os.listdir(self.root))

    def __getitem__(self, index):
        with open(os.path.join(self.root, f"{index}.pkl"), "rb") as f:
            return self.encode(pickle.load(f), index)

    def encode(self, example, index):
        raise NotImplementedError

    @staticmethod
    def collate(batch):
        return default_collate(batch)