import ctypes as ct
from ctypes import CDLL, POINTER
from ctypes import c_size_t, c_int32
import json
import argparse
import sys
import os
import pickle
import math
import argparse
import random
import time
from tqdm import tqdm
import numpy as np
import collections
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import classification_report

from learn_dt import OurDataset, output_scores

MULTIPLIER = 2 ** 16
def get_nth_split(dataset, n_fold, index):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    bottom, top = int(math.floor(float(dataset_size)*index/n_fold)), int(math.floor(float(dataset_size)*(index+1)/n_fold))
    train_indices, test_indices = indices[0:bottom]+indices[top:], indices[bottom:top]
    return train_indices[:opt.maxSize], test_indices[:opt.maxSize]
def get_dataset(data, is_binary):

    data_list = []
    for item in data:
        data_list.append(item)

    new_dataset = []

    for item, y, y2 in tqdm(data_list):
        item = item.numpy()
        y = y.numpy().astype(np.int64)
        y2 = y2.numpy().astype(np.int64)

        average = np.zeros(3, dtype=np.int64)
        deviation = np.zeros(3, dtype=np.int64)
        for i in range(item.shape[0]):
            # item[i,:6]: sport, dport, protocol, tot_len, interval, direction
            current_vector = item[i,:6].astype(np.int64) * MULTIPLIER

            average += current_vector[3:]
            current_average = (average/(i+1)).astype(np.int64)

            deviation += np.abs(current_vector[3:]-current_average)
            current_deviation = (deviation/(i+1)).astype(np.int64)

            final_vector = np.concatenate((current_vector, current_average, current_deviation))
        if is_binary:
            new_dataset.append((final_vector, y[0]))
        else:
            if y2[0] == 0:
                _y = 0
            elif y2[0] == 1 or y2[0] == 2:
                _y = 1
            elif y2[0] >= 3 and y2[0] <= 8:
                _y = 2
            elif y2[0] == 9:
                _y = 3
            elif y2[0] == 10:
                _y = 4
            elif y2[0] == 11 or y2[0] == 12:
                _y = 5
            elif y2[0] == 13 or y2[0] == 14:
                _y = 6
            else:
                _y = 0
            new_dataset.append((final_vector, _y))
    return new_dataset
def ensure_contiguous(array):
    """
    Ensure that array is contiguous
    :param array:
    :return:
    """
    return np.ascontiguousarray(array) if not array.flags['C_CONTIGUOUS'] else array

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--canManipulateBothDirections', action='store_true', help='if the attacker can change packets in both directions of the flow')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--model-path', required=True, help='path to dataset')
    parser.add_argument('--normalizationData', default="", type=str, help='normalization data to use')
    parser.add_argument('--fold', type=int, default=0, help='fold to use')
    parser.add_argument('--nFold', type=int, default=3, help='total number of folds')
    parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
    parser.add_argument('--net', default='', help="path to net (to continue training)")
    parser.add_argument('--function', default='train', help='the function that is going to be called')
    parser.add_argument('--manualSeed', default=0, type=int, help='manual seed')
    parser.add_argument('--maxLength', type=int, default=100, help='max length')
    parser.add_argument('--maxSize', type=int, default=sys.maxsize, help='limit of samples to consider')
#	parser.add_argument("--categoriesMapping", type=str, default="categories_mapping.json", help="mapping of attack categories; see parse.py")
    parser.add_argument('--train_val_split', help='Train validation split ratio', type=float, default=0.8)
    parser.add_argument('--removeChangeable', action='store_true', help='when training remove all features that an attacker could manipulate easily without changing the attack itself')
    parser.add_argument('--tradeoff', type=float, default=0.5, help='max length')
    parser.add_argument('--penaltyTradeoff', type=float, default=0, help='Tradeoff to enforce constant flow duration')
    parser.add_argument('--lr', type=float, default=10**(-2), help='learning rate')
    parser.add_argument('--advTraining', action='store_true', help='Train with adversarial flows')
    parser.add_argument('--allowIATReduction', action='store_true', help='allow reducing IAT below original value')
    parser.add_argument('--order', type=int, default=1, help='order of the norm for adversarial sample generation')
    parser.add_argument('--advMethod', type=str, default="cw", help='which adversarial samples method to use; options are: "cw", "pgd"')
    parser.add_argument('--iterationCount', type=int, default=100, help='number of iterations for creating adversarial samples')
    parser.add_argument('--pathToAdvOutput', type=str, default="", help='path to adv output to be used in pred_plots2')
    parser.add_argument('--averageFeaturesToPruneDuringTraining', type=int, default=-1, help='average number of features that should be "dropped out"; -1 to disable (default)')
    parser.add_argument('--modelSavePeriod', type=float, default=1, help='number of epochs, after which to save the model, can be decimal')
    parser.add_argument('--mutinfo', action='store_true', help='also compute mutinfo during the feature_importance function')
    parser.add_argument('--hidden_size', type=int, default=512, help='number of neurons per layer')
    parser.add_argument('--n_layers', type=int, default=3, help='number of LSTM layers')
    parser.add_argument('--adjustFeatImpDistribution', action='store_true', help='adjust randomization feature importance distributions to a practically relevant shape')
    parser.add_argument('--skipArsDistanceCheck', action='store_true', help='stop ARS computation as soon as 50% theshold is reached')
    parser.add_argument('--is-binary', action='store_true')
    parser.add_argument('--prefix-path', default='/runs', help='the function that is going to be called')

    # parser.add_argument('--nSamples', type=int, default=1, help='number of items to sample for the feature importance metric')

    opt = parser.parse_args()
    print(opt)
    SEED = opt.manualSeed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    with open (opt.dataroot, "rb") as f:
        all_data = pickle.load(f)

    with open(opt.dataroot[:-7]+"_categories_mapping.json", "r") as f:
        categories_mapping_content = json.load(f)
    categories_mapping, mapping = categories_mapping_content["categories_mapping"], categories_mapping_content["mapping"]
    assert min(mapping.values()) == 0

    # a few flows have have invalid IATs due to dataset issues. Sort those out.
    all_data = [item[:opt.maxLength,:] for item in all_data if np.all(item[:,4]>=0)]
    if opt.removeChangeable:
        all_data = [overwrite_manipulable_entries(item) for item in all_data]
    random.shuffle(all_data)
    # print("lens", [len(item) for item in all_data])
    x = [item[:, :-2] for item in all_data]
    y = [item[:, -1:] for item in all_data]
    categories = [item[:, -2:-1] for item in all_data]
    dataset = OurDataset(x, y, categories)
    n_fold = opt.nFold
    fold = opt.fold
    split_r = opt.train_val_split

    train_indices, test_indices = get_nth_split(dataset, n_fold, fold)
    test_data = torch.utils.data.Subset(dataset, test_indices)
    new_dataset_test = get_dataset(test_data, opt.is_binary)
    final_x_test, final_y_test = zip(*new_dataset_test)

    prefix_path = opt.model_path
    with open(f'{prefix_path}/childrenLeft', 'r') as f:
        children_left = np.array(json.load(f))
    with open(f'{prefix_path}/childrenRight', 'r') as f:
        children_right = np.array(json.load(f))
    with open(f'{prefix_path}/threshold', 'r') as f:
        threshold = np.array(json.load(f))
    with open(f'{prefix_path}/feature', 'r') as f:
        feature = np.array(json.load(f))
    with open(f'{prefix_path}/value', 'r') as f:
        value = np.array(json.load(f))

    # value = np.array(value)
    # print(value)
    lib = CDLL("./dt.so")

    dt = lib.dt

    # ND_POINTER_1 = np.ctypeslib.ndpointer(dtype=children_left.dtype, ndim=1, flags="C_CONTIGUOUS")
    children_left_pointer  = children_left.ctypes.data_as(POINTER(ct.c_longlong))
    children_right_pointer = children_right.ctypes.data_as(POINTER(ct.c_longlong))
    threshold_pointer      = threshold.ctypes.data_as(POINTER(ct.c_longlong))
    feature_pointer        = feature.ctypes.data_as(POINTER(ct.c_longlong))
    value_pointer          = value.ctypes.data_as(POINTER(ct.c_longlong))
    c_uint_p = ct.POINTER(ct.c_uint)

    dt.argstypes = [POINTER(ct.c_longlong), POINTER(ct.c_longlong), POINTER(ct.c_longlong), POINTER(ct.c_longlong), POINTER(ct.c_longlong), POINTER(ct.c_longlong), ct.POINTER(ct.c_uint), ct.c_uint]
    # dt.argstypes = [ND_POINTER_1, c_size_t]
    dt.restype = None

    # print(dt(children_left))
    # print(dt_filter(children_left, children_left.size))
    li_elapsed = []
    predictions = []
    N = 1
    for sample in tqdm(final_x_test):
        class_indices = ensure_contiguous(np.zeros(N, dtype=np.uintc))
        shape = class_indices.shape
        class_indices = class_indices.ctypes.data_as(c_uint_p)
        sample = sample.ctypes.data_as(POINTER(ct.c_longlong))
        start = time.perf_counter()
        dt(sample, children_left_pointer, children_right_pointer, value_pointer, feature_pointer, threshold_pointer, class_indices)
        end = time.perf_counter()
        elapsed = end - start
        pred = np.ctypeslib.as_array(class_indices, shape)
        li_elapsed.append(end-start)
        predictions.append(np.array(pred))
    predictions = np.array(predictions)
    li_elapsed = np.array(li_elapsed)
    print(np.mean(li_elapsed), np.std(li_elapsed))
    if opt.is_binary:
        print(classification_report(final_y_test, predictions, labels=[0, 1], digits=6))
    else:
        print(classification_report(final_y_test, predictions, digits=6))
    ret = classification_report(final_y_test, predictions, digits=6, output_dict=True)
    # import pdb; pdb.set_trace()

    with open(f"{opt.model_path}/classification_report_c.json", "w") as f:
        json.dump(ret, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))
    with open(f"{opt.model_path}/inference_time_c.tsv", "w") as f:
        f.write(f"{np.mean(li_elapsed)}\t{np.std(li_elapsed)}")

