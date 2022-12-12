import torch.utils.data as data
import glob
import torch
import numpy as np


class KpiReader(data.Dataset):
    def __init__(self, path):
        super(KpiReader, self).__init__()
        self.path = path
        self.length = len(glob.glob(self.path + '/*.seq'))
        data = []
        for i in range(self.length):
            item = torch.load(self.path+'/%d.seq' % (i+1))
            data.append(item)
        self.data = data

    def __getitem__(self, index):
        kpi_ts, kpi_label, kpi_value = self.data[index]['ts'], self.data[index]['label'], self.data[index]['value']
        return kpi_ts, kpi_label, kpi_value

    def __len__(self):
        return self.length


class LossFunctions:
    eps = 1e-8

    def log_normal(self, x, mu, var):
        """Logarithm of normal distribution with mean=mu and variance=var
           log(x|μ, σ^2) = loss = -0.5 * Σ log(2π) + log(σ^2) + ((x - μ)/σ)^2

        Args:
           x: (array) corresponding array containing the input
           mu: (array) corresponding array containing the mean
           var: (array) corresponding array containing the variance

        Returns:
           output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
        """
        if self.eps > 0.0:
            var = var + self.eps
        return -0.5 * torch.sum(
            np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var)


class ReparameterizeTrick:
    def reparameterize_gaussian(self, mean, logvar, random_sampling=True):
        if random_sampling is True:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5*logvar)
            z = mean + eps*std
            return z
        else:
            return mean