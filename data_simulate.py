import random
import numpy as np
from simulate import gbm_levels
import torch


def process_simulated_data():
    s0 = 10
    mu = np.arange(0.1, 0.15, 0.1)
    mu = np.tile(mu, 10)
    num = 10
    sigma = np.round(np.random.uniform(0.1, 0.55, num), 2)
    paths = 1
    delta = 1.0 / 252.0
    time = 252 * 5
    price_paths = {}
    for i in range(len(sigma)):
        price_paths[i] = {}
        price_paths[i]['prices'] = gbm_levels(s0+i, delta, sigma[i], time, mu[i], paths)
        price_paths[i]['sigma'] = sigma[i]
        price_paths[i]['mu'] = mu[i]

    pad_begin = 29
    all_features = []
    for k_index in price_paths.keys():
        temp_prices = price_paths[k_index]['prices'].squeeze()
        move_prices = []
        for i in range(30):
            move_prices.append(temp_prices[i: len(temp_prices)-30+i])
        move_prices = np.stack(move_prices)
        temp_features = []
        for i in [5, 10, 20, 30, 1]:
            temp_features.append(np.expand_dims(move_prices[-i:].mean(0), 1))
        temp_features.append(np.ones(temp_features[0].shape) * k_index)
        temp_features.append(np.ones(temp_features[0].shape) * price_paths[k_index]['sigma'])
        temp_features.append(np.ones(temp_features[0].shape) * price_paths[k_index]['mu'])
        temp_features = np.concatenate(temp_features, 1)
        price_min = np.min(temp_prices)
        price_max = np.max(temp_prices)
        temp_features[:, :5] = temp_features[:, :5] / price_max
        all_features.append(temp_features)
    all_features = np.stack(all_features)
    return all_features


class MySimuldateDataLoader():
    def __init__(self, device):
        self.device = device
        self.data = process_simulated_data()
        self.train_x = ''
        self.train_y = ''
        self.valid_x = ''
        self.valid_y = ''
        self.test_x = ''
        self.test_y = ''
        self.train_valid_x = ''
        self.train_valid_y = ''
        self.make_data()

    def make_data(self, days = 32, shuffle = True):
        valid_start = self.data.shape[1] - 2 * 252 + 1
        test_start = self.data.shape[1] - 1 * 252 + 1
        temp_data = []
        for i in range(days):
            temp_data.append(np.expand_dims(self.data[:, i:self.data.shape[1]-days+i], 2))
        temp_data = np.concatenate(temp_data, 2)
        train_data = temp_data[:, :valid_start].reshape(-1, days, self.data.shape[-1])
        valid_data = temp_data[:, valid_start:test_start].reshape(-1, days, self.data.shape[-1])
        train_valid_data = temp_data[:, :test_start].reshape(-1, days, self.data.shape[-1])
        test_data = temp_data[:, test_start:].reshape(-1, days, self.data.shape[-1])
        print(train_data.shape, valid_data.shape, test_data.shape)
        if shuffle:
            rand_id = list(range(len(train_data)))
            random.shuffle(rand_id)
            train_data = train_data[rand_id]
            rand_id = list(range(len(train_valid_data)))
            random.shuffle(rand_id)
            train_valid_data = train_valid_data[rand_id]
        self.train_x = torch.Tensor(train_data[:, :, :5]).to(self.device)
        self.train_y = torch.Tensor(train_data[:, 0, 5:]).to(self.device)
        self.valid_x = torch.Tensor(valid_data[:, :, :5]).to(self.device)
        self.valid_y = torch.Tensor(valid_data[:, 0, 5:]).to(self.device)
        self.test_x = torch.Tensor(test_data[:, :, :5]).to(self.device)
        self.test_y = torch.Tensor(test_data[:, 0, 5:]).to(self.device)
        self.train_valid_x = torch.Tensor(train_valid_data[:, :, :5]).to(self.device)
        self.train_valid_y = torch.Tensor(train_valid_data[:, 0, 5:]).to(self.device)


if __name__ == "__main__":
    temp_dataloader = MySimuldateDataLoader()


