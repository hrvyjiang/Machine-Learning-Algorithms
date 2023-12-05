import pandas as pd
import numpy as np
from random import randrange, seed, shuffle
import random


def kfold_cross_validation(data, k, shuffle = False):
    if shuffle:
        fold_size = len(data) // k
        x = list(enumerate(data))
        random.shuffle(x)
        indices, data = zip(*x)
        folds = []
        for i in range(k):
            test_indices = indices[i * fold_size: (i + 1) * fold_size]
            train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]]).astype(np.int64)
            folds.append((train_indices, test_indices))
        return folds
    else:
        fold_size = len(data) // k
        indices = np.arange(len(data))
        folds = []
        for i in range(k):
            test_indices = indices[i * fold_size: (i + 1) * fold_size]
            train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
            folds.append((train_indices, test_indices))
        return folds

# Define the number of folds (K)
k = 10
seed(45)
dataset = [[x] for x in range(1,50)]
fold_indices = kfold_cross_validation(dataset, k, shuffle=False)
fold_indices