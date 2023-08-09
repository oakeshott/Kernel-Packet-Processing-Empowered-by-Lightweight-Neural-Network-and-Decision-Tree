import argparse
import os
import sys
import numpy as np
import random
import pickle
import json
from tqdm import tqdm
import math
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from neural_nets import MLP, ConvNet
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,recall_score, precision_score, f1_score, balanced_accuracy_score

torch.multiprocessing.set_sharing_strategy('file_system')

class PacketDataset(Dataset):
    def __init__(self, data, labels, categories):
        self.data = data
        self.labels = labels
        self.categories = categories
        assert(len(self.data) == len(self.labels) == len(self.categories))

    def __getitem__(self, index):
        data, labels, categories = torch.FloatTensor(self.data[index]), torch.FloatTensor(self.labels[index]), torch.FloatTensor(self.categories[index])
        return data, labels, categories

    def __len__(self):
        return len(self.data)

class PacketFlowDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        assert(len(self.data) == len(self.labels))

    def __getitem__(self, index):
        data, labels = torch.FloatTensor(self.data[index]), torch.FloatTensor(self.labels[index])
        return data, labels

    def __len__(self):
        return len(self.data)

def get_nth_split(dataset, n_fold, index):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    bottom, top = int(math.floor(float(dataset_size)*index/n_fold)), int(math.floor(float(dataset_size)*(index+1)/n_fold))
    train_indices, test_indices = indices[0:bottom]+indices[top:], indices[bottom:top]
    return train_indices, test_indices

def get_dataset(data):
    data_list = []
    for item in data:
        data_list.append(item)

    new_dataset = []
    new_labels = []

    new_dataset2 = []
    for item, y, y2 in tqdm(data_list):
        item = item.numpy()
        y = y.numpy().astype(np.int64)

        average = np.zeros(3)
        deviation = np.zeros(3)
        # average = np.zeros(3, dtype=np.int64)
        # deviation = np.zeros(3, dtype=np.int64)
        for i in range(item.shape[0]):
            # item[i,:6]: sport, dport, protocol, tot_len, interval, direction
            current_vector = item[i,:6]
            # current_vector = item[i,:6].astype(np.int64)
            # current_vector[0] /= 65535
            # current_vector[1] /= 65535
            # current_vector[2] /= 65535
            # current_vector[3] /= 65535
            # current_vector[4] /= 65535
            # if current_vector[2] == 6:
            #     current_vector[2] = 1
            # if current_vector[2] == 13:
            #     current_vector[2] = 0
            # else:
            #     current_vector[2] = 0
            # current_vector[5] /= 65535
            # print(current_vector)
            # current_vector = item[i,:6].astype(np.int64) * MULTIPLIER

            average += current_vector[3:]
            # current_average = (average/(i+1)).astype(np.int64)
            current_average = (average/(i+1))

            deviation += np.abs(current_vector[3:]-current_average)
            current_deviation = (deviation/(i+1))
            # current_deviation = (deviation/(i+1)).astype(np.int64)

            final_vector = np.concatenate((current_vector, current_average, current_deviation))
        new_dataset.append(final_vector)
        # new_dataset2.append(deepcopy(final_vector))
        new_labels.append(y[0])
        # new_dataset.append((final_vector, y[0]))
    scaler = MinMaxScaler()
    scaler.fit(new_dataset)
    new_dataset = scaler.transform(new_dataset)
    # print(scaler.scale_, scaler.data_min_, scaler.data_max_)
    # for i, d in enumerate(new_dataset2):
    #     print(d, ((d - scaler.data_min_)*scaler.scale_*2**16).astype(np.int64), new_dataset[i])
    return new_dataset, new_labels, scaler

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for training a model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hidden_sizes', help='hidden layer dimensions', nargs='+', type=int, default=[16, 16])
    parser.add_argument('--num_epochs', help='number of training epochs', type=int, default=10)
    parser.add_argument('--batch_size', help='batch size', type=int, default=128)
    parser.add_argument('--train_val_split', help='Train validation split ratio', type=float, default=0.8)
    parser.add_argument('--data_dir', help='directory of folder containing the MNIST dataset', default='../dataset')
    # parser.add_argument('--data_dir', help='directory of folder containing the MNIST dataset', default='../data')
    parser.add_argument('--save_dir', help='save directory', default='../saved_models', type=Path)
    parser.add_argument('--maxLength', type=int, default=100, help='max length')
    parser.add_argument('--normalizationData', default="", type=str, help='normalization data to use')
    parser.add_argument('--fold', type=int, default=0, help='fold to use')
    parser.add_argument('--nFold', type=int, default=3, help='total number of folds')
    parser.add_argument('--maxSize', type=int, default=sys.maxsize, help='limit of samples to consider')
    parser.add_argument('--seed', default=0, type=int, help='manual seed')

    args = parser.parse_args()
    print(args.hidden_sizes)
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    with open (os.path.join(args.data_dir, 'flows.pickle'), "rb") as f:
        all_data = pickle.load(f)

    with open(os.path.join(args.data_dir, "flows_categories_mapping.json"), "r") as f:
        categories_mapping_content = json.load(f)

    categories_mapping, mapping = categories_mapping_content["categories_mapping"], categories_mapping_content["mapping"]
    assert min(mapping.values()) == 0

    all_data = [item[:args.maxLength,:] for item in all_data if np.all(item[:,4]>=0)]
    # all_data = [item[:args.maxLength,:] for item in all_data[:10000] if np.all(item[:,4]>=0)]
    random.shuffle(all_data)
    # print("lens", [len(item) for item in all_data])
    x = [item[:, :-2] for item in all_data]
    y = [item[:, -1:] for item in all_data]
    categories = [item[:, -2:-1] for item in all_data]

    dataset = PacketDataset(x, y, categories)

    # split training data to train/validation
    split_r = args.train_val_split

    # globals()[args.function]()
    n_fold = args.nFold
    fold = args.fold

    train_indices, test_indices = get_nth_split(dataset, n_fold, fold)
    train_data = torch.utils.data.Subset(dataset, train_indices)
    test_data = torch.utils.data.Subset(dataset, test_indices)

    new_dataset_train, new_labels_train, transformer = get_dataset(train_data)
    train_trainset = PacketFlowDataset(new_dataset_train, new_labels_train)

    MULTIPLIER = 2 ** 16

    data_min = (transformer.data_min_ * MULTIPLIER).astype(np.int64)
    data_scale = (transformer.scale_ * MULTIPLIER).astype(np.int64)

    with open('data.c', 'w') as f:
        f.write("int64_t data_min[12] = {")
        for i, data in enumerate(data_min):
            if i == len(data_min)-1:
                f.write(f"{data}")
            else:
                f.write(f"{data}, ")
        f.write("};\n")
        f.write("int64_t data_scale[12] = {")
        for i, data in enumerate(data_scale):
            if i == len(data_min)-1:
                f.write(f"{data}")
            else:
                f.write(f"{data}, ")
        f.write("};\n")
    print(data_min, data_scale)
