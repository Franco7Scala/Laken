import torch


class Dataset:

    def __init__(self, input_shape, xs=[], ys=[]):
        self.xs = xs
        self.ys = ys
        self.input_shape = input_shape

    def add_sample(self, x, y):
        self.xs.append(x)
        self.ys.append(y)

    def get_train_dataset(self):
        return torch.reshape(torch.array(self.xs), (len(self.xs), self.input_shape)), torch.array(self.ys)

    def __getitem__(self, index):
        return self.xs[index], self.ys[index], index

    def __len__(self):
        return len(self.xs)
