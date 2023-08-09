"""
Script for running inference of model in C using ctypes
"""
import argparse
import os
import sys
import random
import time
import pickle
import json
from tqdm import tqdm

import numpy as np
import torch
from run_nn import load_c_lib, run_mlp
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,recall_score, precision_score, f1_score, balanced_accuracy_score
from train_mlp import PacketDataset, PacketFlowDataset, get_dataset, get_nth_split

torch.multiprocessing.set_sharing_strategy('file_system')

def output_scores(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    youden = balanced_accuracy_score(y_true, y_pred, adjusted=True)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'Youden']
    print (('{:>11}'*len(metrics)).format(*metrics))
    print ((' {:.8f}'*len(metrics)).format(accuracy, precision, recall, f1, youden))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for testing post-training quantization of a pre-trained model in C",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', help='batch size', type=int, default=1)
    parser.add_argument('--data_dir', help='directory of folder containing the MNIST dataset', default='../dataset')
    parser.add_argument('--maxLength', type=int, default=100, help='max length')
    parser.add_argument('--fold', type=int, default=0, help='fold to use')
    parser.add_argument('--nFold', type=int, default=3, help='total number of folds')
    parser.add_argument('--maxSize', type=int, default=sys.maxsize, help='limit of samples to consider')
    parser.add_argument('--train_val_split', help='Train validation split ratio', type=float, default=0.8)
    parser.add_argument('--seed', default=0, type=int, help='manual seed')
    parser.add_argument('--is-binary', action='store_true')
    parser.add_argument('--save_dir', help='save directory', default='../saved_models')

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

    # all_data = [item[:args.maxLength,:] for item in all_data[:1000000] if np.all(item[:,4]>=0)]
    all_data = [item[:args.maxLength,:] for item in all_data if np.all(item[:,4]>=0)]
    random.shuffle(all_data)
    # print("lens", [len(item) for item in all_data])
    x = [item[:, :-2] for item in all_data]
    y = [item[:, -1:] for item in all_data]
    categories = [item[:, -2:-1] for item in all_data]

    dataset = PacketDataset(x, y, categories)

    split_r = args.train_val_split

    n_fold = args.nFold
    fold = args.fold

    train_indices, test_indices = get_nth_split(dataset, n_fold, fold)
    # train_data = torch.utils.data.Subset(dataset, train_indices)
    test_data = torch.utils.data.Subset(dataset, test_indices)

    # new_dataset_train, new_labels_train = get_dataset(train_data)
    new_dataset_test, new_labels_test = get_dataset(test_data, args.is_binary)
    # train_trainset = PacketFlowDataset(new_dataset_train, new_labels_train)
    test_testset = PacketFlowDataset(new_dataset_test, new_labels_test)
    # train_trainset, train_valset = random_split(train_trainset, [round(len(train_trainset)*split_r), round(len(train_trainset)*(1 - split_r))], generator=torch.Generator().manual_seed(SEED))

    # train_indices, test_indices = train_test_split(list(range(len(dataset.data))), test_size=0.2, stratify=dataset.label, random_state=SEED)
    # test_data = torch.utils.data.Subset(dataset, test_indices)
    # new_dataset_test, new_labels_test = get_dataset(test_data)
    # test_testset = PacketFlowDataset(new_dataset_test, new_labels_test)

    in_dim = 12
    if args.is_binary:
        out_dim = 2
    else:
        out_dim = 7

    print(f'Evaluating integer-only C model on test data')

    test_loader = DataLoader(test_testset, batch_size=1, num_workers=1, shuffle=False)
    # load c library
    c_lib = load_c_lib(library='./mlp.so')

    acc = 0
    all_preds = []
    all_labels = []
    li_elapsed = []
    for samples, labels in tqdm(test_loader):
        samples = (samples * (2 ** 16)).round() # convert to fixed-point 16
        preds, elapsed  = run_mlp(samples, c_lib)
        preds = preds.astype(int)
        li_elapsed.append(elapsed)

        acc += (torch.from_numpy(preds) == labels).sum()

        all_preds += [l for l in preds]
        all_labels += [l for l in labels.detach().numpy().copy()]


    print(f"Accuracy: {(acc / len(test_testset.data)) * 100.0:.2f}%")
    all_labels = np.array(all_labels).squeeze()
    all_preds = np.array(all_preds)
    li_elapsed = np.array(li_elapsed)
    print(np.mean(li_elapsed), np.std(li_elapsed))
    print(classification_report(all_labels, all_preds, digits=6))
    ret = classification_report(all_labels, all_preds, digits=6, output_dict=True)
    with open(f"{args.save_dir}/classification_report_c.json", "w") as f:
        json.dump(ret, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))
    with open(f"{args.save_dir}/inference_time_c.tsv", "w") as f:
        f.write(f"{np.mean(li_elapsed)}\t{np.std(li_elapsed)}")
