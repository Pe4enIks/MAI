import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from model import RBF
from dataset import Dataset

import warnings
warnings.filterwarnings('ignore')


class Pipeline:
    def __init__(self,
                 epochs=1000,
                 lr=0.01,
                 wd=0.0001,
                 save_path='checkpoints/',
                 save_every=500,
                 save_weights=True,
                 logs_path='logs/'):
        self.epochs = epochs
        self.lr = lr
        self.wd = wd
        self.save_path = save_path
        self.logs_path = logs_path
        self.save_every = save_every
        self.save_weights = save_weights

        if not os.path.exists(save_path):
            os.makedirs(save_path, mode=0o777)

        if not os.path.exists(logs_path):
            os.makedirs(logs_path, mode=0o777)

        self.init_dataset()
        self.init_model()

        self.train()
        self.test()

        self.plot()

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.0)

    def func(self, t):
        return np.sin(-5 * t**2 + 10 * t - 5)

    def init_dataset(self):
        start, end, h = 0, 2.5, 0.01

        t = np.linspace(start, end, int((end - start) / h) + 1)
        self.times = t

        dataset = Dataset(t, self.func)

        train_size = int(0.7 * len(dataset))
        valid_size = int(0.2 * len(dataset))

        self.train_dataset = dataset[:train_size]
        self.valid_dataset = dataset[train_size:train_size + valid_size]
        self.test_dataset = dataset[train_size + valid_size:]
        self.plot_dataset = dataset[:]

    def init_model(self):
        self.model = RBF(rbf_features=32, in_features=1, out_features=1)
        self.model.apply(self.init_weights)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.wd)

        self.loss_fn = nn.MSELoss()

    def train(self):
        tqdm_iter = tqdm(range(self.epochs))

        for epoch in tqdm_iter:
            self.model.train()

            train_loss = 0.0
            valid_loss = 0.0

            pred = self.model(self.train_dataset[0].unsqueeze(1))
            target = self.train_dataset[1].unsqueeze(1)

            loss = self.loss_fn(pred, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

            if self.save_weights:
                if epoch % self.save_every == 0 or epoch == self.epochs - 1:
                    self.save(epoch)

            self.model.eval()

            with torch.no_grad():
                pred = self.model(self.valid_dataset[0].unsqueeze(1))
                target = self.valid_dataset[1].unsqueeze(1)

                loss = self.loss_fn(pred, target)
                valid_loss += loss.item()

            tqdm_iter.set_postfix(
                {'epoch:': f'{epoch + 1}/{self.epochs}',
                 'train loss:': train_loss,
                 'valid loss:': valid_loss})

            tqdm_iter.refresh()

    def test(self):
        self.model.eval()

        test_loss = 0.0

        with torch.no_grad():
            pred = self.model(self.test_dataset[0].unsqueeze(1))
            target = self.test_dataset[1].unsqueeze(1)

            loss = self.loss_fn(pred, target)
            test_loss += loss.item()

        print('test loss:', test_loss)

    def save(self, epoch):
        state_dict = self.model.state_dict()

        torch.save(state_dict, self.save_path +
                   f'epoch-{epoch}.pth')

    def plot(self):
        self.model.eval()

        pred = self.model(self.plot_dataset[0].unsqueeze(1))
        target = self.plot_dataset[1]
        x_arr = self.plot_dataset[0]

        pred = pred[:, 0].detach().numpy()

        fig = plt.figure(figsize=(15, 10), dpi=300)

        ax = fig.add_subplot(211)
        ax.grid(True)
        ax.plot(x_arr, target, label='real', color='#0056d6')
        ax.plot(x_arr, pred, label='approx', color='#73d925')
        ax.set_xlabel('time')
        ax.set_ylabel('x')
        ax.legend(loc='upper right')

        plt.savefig(f'{self.logs_path}function.png')


def main():
    Pipeline()


if __name__ == '__main__':
    main()
