from torch.utils.data import Dataset

import os
import pickle
import numpy as np



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

class LazyLatticeDataset(LazyDataset):

    def __init__(self, root,
                 input_len=None,
                 num_labels=None,
                 start_token=None,
                 clf_token=None,
                 cap_length=None):
        super(LazyLatticeDataset, self).__init__(root)
        self.input_len = input_len
        self.num_labels = num_labels
        self.start_token = start_token
        self.clf_token = clf_token
        self.cap_length = cap_length

    def encode(self, example, index):
        input_ids = np.zeros((self.input_len,), dtype=np.int64)
        pos_ids = np.zeros((self.input_len,), dtype=np.int64)
        labels = np.zeros((self.num_labels), dtype=np.float32)

        # input ids
        sent = [self.start_token] + example["input_ids"][:self.cap_length] + [self.clf_token]  # It's already integer ids now
        if None in sent:
            print(sent)
            print(example["text"])
            for lattice in example["lattice"]:
                print(lattice)
        input_ids[:len(sent)] = sent

        # position ids
        print(example["pos_ids"], self.cap_length)
        raw_pos = example["pos_ids"][:self.cap_length]
        padded_pos = [0] + [pos + 1 for pos in raw_pos] + [max(raw_pos) + 2]
        pos_ids[:len(padded_pos)] = padded_pos

        # classification ids
        token_ids = len(sent) - 1  # classification token? Yes

        # labels
        for l in example["labels"]:
            labels[l] = 1.0
        return (input_ids, token_ids, pos_ids, labels, example["text"], example["lattice"])


def cumsum(l):
    # returns cumulitive sums from left to right,
    # note that output will have extra zero at the left side
    out = [0]
    for num in l:
        out.append(out[-1] + num)
    return out

class LazyMLMDataset(LazyDataset):

    def __init__(self, root,
                 input_len=None,
                 num_labels=None,
                 start_token=None,
                 end_token=None,
                 mask_token=None,
                 cap_length=None,
                 mask_ratio=0.15,
                 seethrough=False):
        super(LazyMLMDataset, self).__init__(root)
        self.input_len = input_len
        self.num_labels = num_labels
        self.start_token = start_token
        self.end_token = end_token
        self.cap_length = cap_length
        self.mask_ratio = mask_ratio
        self.mask_token = mask_token
        self.seethrough = seethrough

    def encode(self, example, index):
        return self.encode_static(example)

    def encode_static(self, example):
        input_ids = np.zeros((self.input_len,), dtype=np.int64)
        pos_ids = np.zeros((self.input_len,), dtype=np.int64)
        labels = np.ones((self.input_len,), dtype=np.int64) * -100

        # input ids
        sent = [self.start_token] + example["input_ids"][:self.cap_length] + [
            self.end_token]  # It's already integer ids now
        sent_labels = [self.start_token] + example["labels"][:self.cap_length] + [
            self.end_token]  # It's already integer ids now
        input_ids[:len(sent)] = sent
        labels[:len(sent)] = sent_labels

        # position ids
        raw_pos = example["pos_ids"][:self.cap_length]
        padded_pos = [0] + [pos + 1 for pos in raw_pos] + [max(raw_pos) + 2]
        pos_ids[:len(padded_pos)] = padded_pos

        return (input_ids, pos_ids, labels, example["text"], example["lattice"])

class LazyMLMTrainDataset(LazyMLMDataset):

    def __init__(self, *args, **kwargs):
        super(LazyMLMTrainDataset, self).__init__(*args, **kwargs)
        self.stochastic = True

    def encode(self, example, index):
        (input_ids, pos_ids, labels, text, lattice) = self.encode_static(example)
        if self.stochastic:
            input_ids, labels = self.random_mask(input_ids, labels, lattice, text)
        return (input_ids, pos_ids, labels, text, lattice)

    def random_mask(self, input_ids, labels, lattice, text):
        sizes = [len(edges_of(chunk)) for chunk in lattice]
        raw_alignment = cumsum(sizes)[:-1]
        shift = 1  # from adding BOS at the start
        actual_choices = ((np.array(raw_alignment) + shift) < len(input_ids)).sum()
        sizes = sizes[:actual_choices]
        raw_alignment = raw_alignment[:actual_choices]
        mask_number = math.ceil(self.mask_ratio * actual_choices)  # mask whole chunks
        mask_chunk_ids = random.sample(list(range(actual_choices)), k=mask_number)
        # print([(text.split(" ")[i], labels[raw_alignment[i] + shift]) for i in mask_chunk_ids])
        label_mask = np.ones(labels.shape, dtype=np.bool_)
        input_mask = np.zeros(input_ids.shape, dtype=np.bool_)
        for id in mask_chunk_ids:
            s = raw_alignment[id] + shift
            label_mask[s] = False
            input_mask[s: s + sizes[id]] = True
        if not self.seethrough:
            new_input_ids = np.ma.array(input_ids, mask=input_mask).filled(fill_value=self.mask_token)
        else:
            new_input_ids = input_ids
        new_labels = np.ma.array(labels, mask=label_mask).filled(fill_value=-100)
        # print(new_labels)
        if (new_labels !=-100).sum() != len(mask_chunk_ids):
            raise ValueError
        return new_input_ids, new_labels

class LazyMLMEvalDataset(LazyMLMTrainDataset):

    def encode(self, example, index):
        (input_ids, pos_ids, labels, text, lattice) = self.encode_static(example)
        if self.stochastic:
            random.seed(index)
            input_ids, labels = self.random_mask(input_ids, labels, lattice, text)
        return (input_ids, pos_ids, labels, text, lattice)

class LazyMLMFeaturePredictionDataset(LazyDataset):

    def __init__(self, root,
                 input_len=None,
                 num_labels=None,
                 start_token=None,
                 end_token=None,
                 mask_token=None,
                 cap_length=None,
                 mask_ratio=0.15,
                 seethrough=False,
                 num_features=3,
                 feature_tokens=None):
        super(LazyMLMFeaturePredictionDataset, self).__init__(root)
        self.input_len = input_len
        self.num_labels = num_labels
        self.start_token = start_token
        self.end_token = end_token
        self.cap_length = cap_length
        self.mask_ratio = mask_ratio
        self.mask_token = mask_token
        self.seethrough = seethrough
        self.num_features = num_features
        self.feature_tokens = feature_tokens

    def encode(self, example, index):
        return self.encode_static(example)

    def encode_static(self, example):
        input_ids = np.zeros((self.input_len,), dtype=np.int64)
        pos_ids = np.zeros((self.input_len,), dtype=np.int64)
        labels = np.ones((self.input_len,), dtype=np.int64) * -100

        # input ids
        sent = [self.start_token] + example["input_ids"][:self.cap_length] + [self.end_token]  # It's already integer ids now
        sent_labels = [-100] + example["features"] + [-100] * (len(example["input_ids"][len(example["features"]):self.cap_length]))+ [-100]  # It's already integer ids now
        input_ids[:len(sent)] = sent
        labels[:len(sent)] = sent_labels

        # position ids
        raw_pos = example["pos_ids"][:self.cap_length]
        padded_pos = list(range(self.num_features + 1)) + [pos + self.num_features + 1 for pos in raw_pos] + [max(raw_pos) +  self.num_features + 1 + 1]
        pos_ids[:len(padded_pos)] = padded_pos
        return (input_ids, pos_ids, labels, example["text"], example["lattice"])
