"""
Script for training a simple MLP for classification on the MNIST dataset
"""
import argparse
import os
import sys
import numpy as np
import random
import pickle
import json
import time
from tqdm import tqdm
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from neural_nets import MLP, ConvNet
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,recall_score, precision_score, f1_score, balanced_accuracy_score
from train_mlp import PacketDataset, PacketFlowDataset, get_dataset, get_nth_split, get_dataset, output_scores

torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for training a model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hidden_sizes', help='hidden layer dimensions', nargs='+', type=int, default=[8, 8])
    parser.add_argument('--filename', help='filename', type=str, default='mlp_pktflw.th')
    parser.add_argument('--num_epochs', help='number of training epochs', type=int, default=10)
    parser.add_argument('--batch_size', help='batch size', type=int, default=128)
    parser.add_argument('--train_val_split', help='Train validation split ratio', type=float, default=0.8)
    parser.add_argument('--data_dir', help='directory of folder containing the MNIST dataset', default='../dataset')
    # parser.add_argument('--data_dir', help='directory of folder containing the MNIST dataset', default='../data')
    parser.add_argument('--save_dir', help='save directory', default='../saved_models')
    parser.add_argument('--maxLength', type=int, default=100, help='max length')
    parser.add_argument('--normalizationData', default="", type=str, help='normalization data to use')
    parser.add_argument('--fold', type=int, default=0, help='fold to use')
    parser.add_argument('--nFold', type=int, default=3, help='total number of folds')
    parser.add_argument('--maxSize', type=int, default=sys.maxsize, help='limit of samples to consider')
    parser.add_argument('--seed', default=0, type=int, help='manual seed')
    parser.add_argument('--is-binary', action='store_true')

    args = parser.parse_args()
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

    # new_dataset_train, new_labels_train = get_dataset(train_data)
    new_dataset_test, new_labels_test = get_dataset(test_data, args.is_binary)
    # train_trainset = PacketFlowDataset(new_dataset_train, new_labels_train)
    test_testset = PacketFlowDataset(new_dataset_test, new_labels_test)
    # train_trainset, train_valset = random_split(train_trainset, [round(len(train_trainset)*split_r), round(len(train_trainset)*(1 - split_r))], generator=torch.Generator().manual_seed(SEED))

    # new_dataset, new_labels = get_dataset(dataset)
    # dataset = PacketFlowDataset(new_dataset, new_labels)
    # train_indices, test_indices = train_test_split(list(range(len(dataset))), test_size=0.2, stratify=dataset.labels, random_state=SEED)
    # test_testset = torch.utils.data.Subset(dataset, test_indices)
    # train_indices, val_indices = train_test_split(train_indices, test_size=0.2, stratify=np.array(dataset.labels)[train_indices], random_state=SEED)
    # train_trainset = torch.utils.data.Subset(dataset, train_indices)
    # train_valset = torch.utils.data.Subset(dataset, val_indices)
    saved_stats = torch.load(os.path.join(args.save_dir, args.filename))

    state_dict = saved_stats['state_dict']


    in_dim = 12
    if args.is_binary:
        out_dim = 2
    else:
        out_dim = 7
    model = MLP(in_dim=in_dim, out_dim=out_dim, hidden_sizes=args.hidden_sizes)
    model.load_state_dict(state_dict)

    # optimizer = Adam(model.parameters(), lr=1e-3)
    #
    # loss_fnc = nn.CrossEntropyLoss()
    # loss_fnc = nn.BCELoss()

    # train_loader = DataLoader(train_trainset, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(train_valset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_testset, batch_size=1, shuffle=False)

    print('Evaluate model on test data')
    model.eval()
    all_preds = []
    all_labels = []
    li_elapsed = []
    with torch.no_grad():
        acc = 0
        for samples, labels in tqdm(test_loader):
            start = time.perf_counter()
            logits = model(samples.float())
            probs = torch.nn.functional.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            end = time.perf_counter()
            acc += (preds == labels).sum()
            all_labels += [label for label in labels.detach().numpy().copy()]
            all_preds += [pred for pred in preds.detach().numpy().copy()]
            elapsed = end - start
            li_elapsed.append(elapsed)

    li_elapsed = np.array(li_elapsed)
    print(np.mean(li_elapsed), np.std(li_elapsed))
    print(f"Accuracy: {(acc / len(test_testset))*100.0:.3f}%")
    # os.makedirs(args.save_dir, exist_ok=True)
    # torch.save({'state_dict': model.state_dict(),
    #             'hidden_sizes': args.hidden_sizes,
    #             'train_loss': train_loss,
    #             'val_loss': val_loss,
    #             'test_acc': acc},
    #             f"{args.save_dir}/mlp_pktflw.th")
    all_labels = np.array(all_labels).squeeze()
    all_preds = np.array(all_preds)
    for i, label in enumerate(all_labels):
        if label == 6:
            print(i, all_preds[i], label)

    print(classification_report(all_labels, all_preds, digits=6))
    ret = classification_report(all_labels, all_preds, digits=6, output_dict=True)
    with open(f"{args.save_dir}/classification_report_torch.json", "w") as f:
        json.dump(ret, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))
    with open(f"{args.save_dir}/inference_time_torch.tsv", "w") as f:
        f.write(f"{np.mean(li_elapsed)}\t{np.std(li_elapsed)}")
