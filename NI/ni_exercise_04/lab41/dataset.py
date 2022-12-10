from torch.utils.data import Dataset as Base
import torch
import numpy as np


class Dataset(Base):
    def __init__(self, params_dct):
        super().__init__()
        params_elips1 = params_dct['elipsis1']
        params_elips2 = params_dct['elipsis2']
        params_parabola = params_dct['parabola']

        self.elips1 = self.gen_elipsis(params_elips1['a'],
                                       params_elips1['b'],
                                       params_elips1['alpha'],
                                       params_elips1['x0'],
                                       params_elips1['y0'])

        self.elips2 = self.gen_elipsis(params_elips2['a'],
                                       params_elips2['b'],
                                       params_elips2['alpha'],
                                       params_elips2['x0'],
                                       params_elips2['y0'])

        self.parab = self.gen_parabola(params_parabola['p'],
                                       params_parabola['alpha'],
                                       params_parabola['x0'],
                                       params_parabola['y0'])

        self.samples, self.labels = self.concat_tensors(self.elips1,
                                                        self.elips2,
                                                        self.parab)

        self.samples, self.labels = self.permute(self.samples, self.labels)

    def elipsis(self, a, b):
        start = 0.0
        end = 2 * np.pi
        step = 0.025
        vals = np.linspace(start, end, int((end - start) / step) + 1)
        x = np.array([a * np.cos(t) for t in vals])
        y = np.array([b * np.sin(t) for t in vals])
        return x, y

    def parabola(self, p):
        start = -1.0
        end = 1.0
        step = 0.025
        x = np.linspace(start, end, int((end - start) / step) + 1)
        y = np.array([(el ** 2) / (2 * p) for el in x])
        return x, y

    def func1(self, x, y, alpha):
        return np.cos(alpha) * x + np.sin(alpha) * y

    def func2(self, x, y, alpha):
        return -np.sin(alpha) * x + np.cos(alpha) * y

    def rotate_coords(self, coords, alpha, x0, y0):
        res_x, res_y = [], []

        for x, y in zip(coords[0], coords[1]):
            res_x.append(self.func1(x, y, alpha) + x0)
            res_y.append(self.func2(x, y, alpha) + y0)

        x = torch.FloatTensor(np.array(res_x))
        y = torch.FloatTensor(np.array(res_y))

        return x, y

    def gen_elipsis(self, a, b, alpha, x0, y0):
        elipsis = self.elipsis(a, b)
        x, y = self.rotate_coords(elipsis, alpha, x0, y0)
        return x, y

    def gen_parabola(self, p, alpha, x0, y0):
        parabola = self.parabola(p)
        x, y = self.rotate_coords(parabola, alpha, x0, y0)
        return x, y

    def concat_tensors(self, t1, t2, t3):
        x = torch.cat((t1[0], t2[0], t3[0]), dim=0)
        y = torch.cat((t1[1], t2[1], t3[1]), dim=0)
        classes = []

        samples = torch.FloatTensor(np.vstack((np.array(x), np.array(y))))

        first_class = len(t1[0])
        second_class = len(t1[0]) + len(t2[0])
        third_class = len(t1[0]) + len(t2[0]) + len(t3[0])

        for i in range(samples.shape[1]):
            if i < first_class:
                classes.append(0)
            elif i < second_class:
                classes.append(1)
            elif i < third_class:
                classes.append(2)

        samples = torch.permute(samples, (1, 0))
        targets = torch.LongTensor(classes)
        return samples, targets

    def permute(self, samples, labels):
        permut = np.random.permutation(samples.shape[0])
        return samples[permut], labels[permut]

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]
