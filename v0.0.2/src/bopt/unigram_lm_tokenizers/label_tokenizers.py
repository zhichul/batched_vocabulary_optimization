from typing import List

import torch


class LatticeLabelTokenizer:

    def __init__(self, vocabulary, padding=-100):
        self.vocbulary = vocabulary
        self.device = "cpu"
        self.padding = padding

    def to(self, device):
        self.device = device

    def __call__(self, labels: List[List[str]], M, KNE, memoizer=None, ids=None):
        # TODO: This assumes L is large enough that the number of labels all get offset by M
        # This also assumes the start_position based linearization order
        outputs = []
        for i, label in enumerate(labels):
            if memoizer is None or (ids[i] not in memoizer):
                label_ids = torch.ones((KNE,), dtype=torch.long).fill_(self.padding)
                for j, label_item in enumerate(label):
                    label_ids[j * M] = self.vocbulary.index(label_item)
                outputs.append(label_ids)
                if memoizer:
                    memoizer[ids[i]] = label_ids
            else:
                outputs.append(memoizer[ids[i]])
        return torch.stack(outputs, dim=0).to(self.device)

    def retrieve_predictions(self, predictions_tensor, label_tensor):
        """
        Assumes every row has the same number of predictions
        """
        return (predictions_tensor[label_tensor != self.padding].reshape(*(label_tensor.size()[:-1]+(-1,))),
                label_tensor[label_tensor != self.padding].reshape(*(label_tensor.size()[:-1] + (-1,))))

class NBestLabelTokenizer(LatticeLabelTokenizer):

    def __call__(self, labels: List[List[str]], seq_length, memoizer=None, ids=None):
        # This also assumes the start_position based linearization order
        outputs = []
        for i, label in enumerate(labels):
            if memoizer is None or (ids[i] not in memoizer):
                label_ids = torch.ones((seq_length,), dtype=torch.long).fill_(self.padding)
                for j, label_item in enumerate(label):
                    label_ids[j] = self.vocbulary.index(label_item)
                outputs.append(label_ids)
                if memoizer:
                    memoizer[ids[i]] = label_ids
            else:
                outputs.append(memoizer[ids[i]])
        return torch.stack(outputs, dim=0).to(self.device)