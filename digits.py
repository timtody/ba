from scipy.io import loadmat
import numpy as np
from math import ceil
import time


class Digits:
    def __init__(self, path, split=.98):
        self.split = split
        mat = loadmat(path)
        ix = int(mat["images"].shape[0]*split)
        self.train = Train(images=mat["images"][:ix], labels=mat["targets"][:ix])
        self.test = Test(images=mat["images"][ix:], labels=mat["targets"][ix:])
        print("Training size: {}, Test Size: {}".format(len(self.train.images), len(self.test.images)))


class Train:
    pt = 0
    def __init__(self, images, labels):
        self.images = images.reshape(-1, 1024) / 255
        self.labels = list(map(lambda x: [1 if i == x else 0 for i in range(10)], labels))

    def next_batch(self, n):
        self.pt += n
        if self.pt > len(self.images):
            self.pt = n
        out = self.images[self.pt-n:self.pt], self.labels[self.pt-n:self.pt]
        return out


class Test:
    def __init__(self, images, labels):
        self.images = images.reshape(-1, 1024) / 255
        self.labels = list(map(lambda x: [1 if i == x else 0 for i in range(10)], labels))
        print(len(self.images), len(self.labels))

