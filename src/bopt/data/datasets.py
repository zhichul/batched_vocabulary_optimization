import json

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

class LazySkipGramDataset(Dataset):

    def __init__(self, root, max_block_length):
        self.root = root
        with open(f"{root}.index.json", "rt") as f:
            index = json.load(f)
        self.length = index["total"]
        self.counts = index["counts"]
        self.words = index["words"]
        self.w2i = {word:i for i, word in enumerate(self.words)}
        _index = [((dist, self.w2i[src], self.w2i[tgt]), count)
                       for dist, dist_count in self.counts.items()
                       for src, tgt_count in dist_count.items()
                       for tgt, count in tgt_count.items()]
        self._index = [content for (content, count) in _index for _ in range(count)]
        self.max_block_length = max_block_length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        dist, src, tgt = self._index[index]
        with open(os.path.join(self.root, f"{src}.pkl"), "rb") as f:
            with open(os.path.join(self.root, f"{tgt}.pkl"), "rb") as g:
                return self.encode({"src": pickle.load(f), "tgt": pickle.load(g)}, (index, dist, src, tgt, self.max_block_length))

    def encode(self, example, index):
        raise NotImplementedError

    @staticmethod
    def collate(batch):
        return default_collate(batch)