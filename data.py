import random
import numpy as np
import pickle as pkl
import json
import os
from datetime import datetime
import torch


def process_eod_data(market_name):
    data_path = os.path.join(os.getcwd(), './data')
    date_format = '%Y-%m-%d %H:%M:%S'
    begin_date = datetime.strptime('2012-11-19 00:00:00', date_format)
    return_days = 1
    pad_begin = 29

    trading_dates = np.genfromtxt(
        os.path.join(data_path, market_name + '_aver_line_dates.csv'), dtype=str, delimiter=',', skip_header=False)
    # the data ends at 2017
    index_tra_dates = {}
    tra_dates_index = {}
    for index, date in enumerate(trading_dates):
        tra_dates_index[date] = index
        index_tra_dates[index] = date
    tickers = np.genfromtxt(
        os.path.join(data_path, market_name + '_tickers_qualify_dr-0.98_min-5_smooth.csv'),
        dtype=str, delimiter='\t', skip_header=False)
    print('#tickers selected:', len(tickers))

    data_EOD = []
    for index, ticker in enumerate(tickers):
        single_EOD = np.genfromtxt(
            os.path.join(data_path, 'google_finance', market_name + '_' + ticker + '_30Y.csv'),
            dtype=str, delimiter=',', skip_header=True)
        data_EOD.append(single_EOD)

    all_features = {}

    for stock_index, single_EOD in enumerate(data_EOD):
        # select data within the begin_date
        begin_date_row = -1
        for date_index, daily_EOD in enumerate(single_EOD):
            date_str = daily_EOD[0].replace('-05:00', '')
            date_str = date_str.replace('-04:00', '')
            cur_date = datetime.strptime(date_str, date_format)
            if cur_date > begin_date:
                begin_date_row = date_index
                break
        selected_EOD_str = single_EOD[begin_date_row:]
        selected_EOD = np.zeros(selected_EOD_str.shape, dtype=float)
        for row, daily_EOD in enumerate(selected_EOD_str):
            date_str = daily_EOD[0].replace('-05:00', '')
            date_str = date_str.replace('-04:00', '')
            selected_EOD[row][0] = tra_dates_index[date_str]
            for col in range(1, selected_EOD_str.shape[1]):
                selected_EOD[row][col] = float(daily_EOD[col])

        # calculate moving average features
        begin_date_row = -1
        for row in selected_EOD[:, 0]:
            row = int(row)
            if row >= pad_begin:  # offset for the first 30-days average
                begin_date_row = row
                break
        mov_aver_features = np.zeros([selected_EOD.shape[0], 4 * 5], dtype=float)
        # 4 columns refers to 5-, 10-, 20-, 30-days average for 5 prices Open, High, Low, Close, Volume
        # 5-Open, 10-Open, 20-Open, 30-Open, 5-High, 10-High, 20-High, 30-High, ...
        for row in range(begin_date_row, selected_EOD.shape[0]):
            date_index = selected_EOD[row][0]
            aver_5, aver_10, aver_20, aver_30 = [0.0] * 5, [0.0] * 5, [0.0] * 5, [0.0] * 5
            count_5, count_10, count_20, count_30 = 0, 0, 0, 0
            for offset in range(30):
                date_gap = date_index - selected_EOD[row - offset][0]
                if date_gap < 5:
                    count_5 += 1
                    for price_index in range(1, 6):
                        aver_5[price_index-1] += selected_EOD[row - offset][price_index]
                if date_gap < 10:
                    count_10 += 1
                    for price_index in range(1, 6):
                        aver_10[price_index-1] += selected_EOD[row - offset][price_index]
                if date_gap < 20:
                    count_20 += 1
                    for price_index in range(1, 6):
                        aver_20[price_index-1] += selected_EOD[row - offset][price_index]
                if date_gap < 30:
                    count_30 += 1
                    for price_index in range(1, 6):
                        aver_30[price_index-1] += selected_EOD[row - offset][price_index]
            for price_index in range(5):
                mov_aver_features[row][4 * price_index + 0] = aver_5[price_index] / count_5
            for price_index in range(5):
                mov_aver_features[row][4 * price_index + 1] = aver_10[price_index] / count_10
            for price_index in range(5):
                mov_aver_features[row][4 * price_index + 2] = aver_20[price_index] / count_20
            for price_index in range(5):
                mov_aver_features[row][4 * price_index + 3] = aver_30[price_index] / count_30


        '''
        normalize features by feature / max, the max price is the
        max of relevant price values, I give up to subtract min for easier
        return ratio calculation. (following previous works)
        '''
        price_min = np.min(selected_EOD[begin_date_row:, 1:6], 0)
        price_max = np.max(selected_EOD[begin_date_row:, 1:6], 0)
        for price_index in range(5):
            mov_aver_features[:, 4 * price_index: 4 * price_index + 4] = (
                    mov_aver_features[:, 4 * price_index: 4 * price_index + 4] / price_max[price_index])

        # 5 columns refers to 5-, 10-, 20-, 30-days average, ori_value for 5 prices Open, High, Low, Close, Volume
        # 5-Open, 10-Open, 20-Open, 30-Open, Open, 5-High, 10-High, 20-High, 30-High, High, ...
        features = np.ones([len(trading_dates) - pad_begin, 1+5*5], dtype=float) * -1234
        # data missed at the beginning
        for row in range(len(trading_dates) - pad_begin):
            features[row][0] = row
        for row in range(begin_date_row, selected_EOD.shape[0]):
            cur_index = int(selected_EOD[row][0])
            for price_index in range(5):
                features[cur_index - pad_begin][1+5*price_index: 1+5*price_index+4] \
                    = mov_aver_features[row][4*price_index: 4*price_index+4]
                if cur_index - int(selected_EOD[row - return_days][0]) == return_days:
                    features[cur_index - pad_begin][1+5*price_index+4] \
                        = selected_EOD[row][1+price_index] / price_max[price_index]
        all_features[tickers[stock_index]] = features

    save_data = {'all_features': all_features, 'index_tra_dates': index_tra_dates, 'tra_dates_index': tra_dates_index}
    with open(os.path.join(data_path, market_name + '_all_features.pkl'), 'wb') as fw:
        pkl.dump(save_data, fw)


def extract_sector_data(market_name):
    data_path = './data'
    with open('./data/' + market_name + '_details.pkl', 'rb') as fr:
        data = pkl.load(fr)
    industries = {}
    with open(os.path.join(data_path, market_name + '_list'), 'r') as fr:
        stockcodes = [symbol.strip() for symbol in fr.readlines()]
    for k in stockcodes:
        try:
            if data[k]['sector'] not in industries.keys():
                industries[data[k]['sector']] = [k]
            else:
                industries[data[k]['sector']].append(k)
        except:
            print(k)
    with open(os.path.join('./data', 'relation/sector_industry',
                           market_name + '_sector_ticker.json'), 'w') as fw:
        json.dump(industries, fw)
    return


class MyDataLoader():
    def __init__(self, args, market_name, seq_len, feature_describe, save_memory, device):
        self.args = args
        self.save_memory = save_memory
        self.device = device
        self.index_tra_dates = None
        self.tra_dates_index = None
        if market_name == 'NASDAQ' or market_name == 'NYSE':
            self.valid_start_index = 756
            self.test_start_index = 1008
        elif market_name == 'NASDAQ2':
            self.valid_start_index = 757
            self.test_start_index = 1008
        elif market_name == 'topix100':
            self.valid_start_index = 690
            self.test_start_index = 938
        elif market_name == 'ftse100':
            self.valid_start_index = 759
            self.test_start_index = 1012
        else:
            print('Invalid market')
        stocksl_data_file = ('./data/stocksl_data' + '_market-' + market_name + '_days-' + str(seq_len)
                             + '_feauture-' + feature_describe + '_seed-' + str(args.seed)+'.pkl')
        print('Making stocksl data')
        train_x, train_y, train_mask, valid_x, valid_y, valid_mask, test_x, test_y, test_mask \
            = self.make_stocksel_dataset(market_name, seq_len, feature_describe)
        if self.save_memory:
            device = 'cpu'
        self.train_x = torch.Tensor(train_x).to(device) # NASDAQ [740, 1026, 16, 5]
        self.train_y = torch.Tensor(train_y).to(device) # NASDAQ [740, 1026, 2]
        self.train_mask = torch.Tensor(train_mask).to(device) # NASDAQ [740, 1026]
        self.valid_x = torch.Tensor(valid_x).to(device) # NASDAQ [252, 1026, 16, 5]
        self.valid_y = torch.Tensor(valid_y).to(device) # NASDAQ [252, 1026, 2]
        self.valid_mask = torch.Tensor(valid_mask).to(device) # NASDAQ [252, 1026]
        self.test_x = torch.Tensor(test_x).to(device) # NASDAQ [237, 1026, 16, 5]
        self.test_y = torch.Tensor(test_y).to(device) # NASDAQ [237, 1026, 2]
        self.test_mask = torch.Tensor(test_mask).to(device) # NASDAQ [237, 1026]

        self.batch_x = ''
        self.batch_y = ''
        self.batch_mask = ''

    def move_device(self):
        self.batch_x = self.batch_x.to(self.device)
        self.batch_y = self.batch_y.to(self.device)
        self.batch_mask = self.batch_mask.to(self.device)

    def get_train_data(self, shuffle=False):
        self.batch_x, self.batch_y, self.batch_mask = '', '', ''
        if not shuffle:
            self.batch_x = self.train_x
            self.batch_y = self.train_y
            self.batch_mask = self.train_mask
        else:
            rand_id = list(range(len(self.train_x)))
            random.shuffle(rand_id)
            self.batch_x = self.train_x[rand_id]
            self.batch_y = self.train_y[rand_id]
            self.batch_mask = self.train_mask[rand_id]
        if self.save_memory:
            self.move_device()

    def get_valid_data(self, shuffle=False):
        self.batch_x, self.batch_y, self.batch_mask = '', '', ''
        if not shuffle:
            self.batch_x = self.valid_x
            self.batch_y = self.valid_y
            self.batch_mask = self.valid_mask
        else:
            rand_id = list(range(len(self.valid_x)))
            random.shuffle(rand_id)
            self.batch_x = self.valid_x[rand_id]
            self.batch_y = self.valid_y[rand_id]
            self.batch_mask = self.valid_mask[rand_id]
        if self.save_memory:
            self.move_device()

    def get_test_data(self, shuffle=False):
        self.batch_x, self.batch_y, self.batch_mask = '', '', ''
        if not shuffle:
            self.batch_x = self.test_x
            self.batch_y = self.test_y
            self.batch_mask = self.test_mask
        else:
            rand_id = list(range(len(self.test_x)))
            random.shuffle(rand_id)
            self.batch_x = self.test_x[rand_id]
            self.batch_y = self.test_y[rand_id]
            self.batch_mask = self.test_mask[rand_id]
        if self.save_memory:
            self.move_device()

    def get_train_valid_data(self, shuffle=False):
        self.batch_x, self.batch_y, self.batch_mask = '', '', ''
        train_valid_x = torch.cat((self.train_x, self.valid_x), 0)
        train_valid_y = torch.cat((self.train_y, self.valid_y), 0)
        train_valid_mask = torch.cat((self.train_mask, self.valid_mask), 0)
        if not shuffle:
            self.batch_x = train_valid_x
            self.batch_y = train_valid_y
            self.batch_mask = train_valid_mask
        else:
            rand_id = list(range(len(train_valid_x)))
            random.shuffle(rand_id)
            self.batch_x = train_valid_x[rand_id]
            self.batch_y = train_valid_y[rand_id]
            self.batch_mask = train_valid_mask[rand_id]
        if self.save_memory:
            self.move_device()

    def make_stocksel_dataset(self, market_name, seq_len, feature_describe):
        data_path = os.path.join(os.getcwd(), './data')
        with open(os.path.join(data_path, market_name + '_all_features.pkl'), 'rb') as fr:
            data = pkl.load(fr)
        all_stock_features = data['all_features']

        all_features = [all_stock_features[stock][:-1, 1:] for stock in all_stock_features.keys()]
        all_features = np.stack(all_features)
        # 5 prices Open, High, Low, Close, Volume
        # 5-Open, 10-Open, 20-Open, 30-Open, 5-High, 10-High, 20-High, 30-High, ...

        # the index is for y
        valid_start_index = self.valid_start_index
        test_start_index = self.test_start_index
        label_dim = 19
        if feature_describe == 'all':
            feature_dim = list(range(25))
        elif feature_describe == 'close_only':
            feature_dim = [15, 16, 17, 18, 19]
        else:
            feature_dim = feature_describe
        train_x, train_y, valid_x, valid_y, test_x, test_y = [], [], [], [], [], []
        for i in range(seq_len, all_features.shape[1]):
            if i < valid_start_index:
                train_x.append(all_features[:, i-seq_len: i, feature_dim])
                train_y.append(all_features[:, i-1: i+1, label_dim])
            elif i < test_start_index:
                valid_x.append(all_features[:, i-seq_len: i, feature_dim])
                valid_y.append(all_features[:, i-1: i+1, label_dim])
            else:
                test_x.append(all_features[:, i - seq_len: i, feature_dim])
                test_y.append(all_features[:, i-1: i+1, label_dim])
        train_x = np.stack(train_x)
        train_y = np.stack(train_y)
        valid_x = np.stack(valid_x)
        valid_y = np.stack(valid_y)
        test_x = np.stack(test_x)
        test_y = np.stack(test_y)

        train_mask = ((train_x == -1234).sum(3).sum(2) + (train_y == -1234).sum(2) == 0).astype(float)
        valid_mask = ((valid_x == -1234).sum(3).sum(2) + (valid_y == -1234).sum(2) == 0).astype(float)
        test_mask = ((test_x == -1234).sum(3).sum(2) + (test_y == -1234).sum(2) == 0).astype(float)
        return train_x, train_y, train_mask, valid_x, valid_y, valid_mask, test_x, test_y, test_mask


class MyPretrainDataLoader():
    def __init__(self, args, market_name, seq_len, feature_describe, save_memory, device):
        self.args = args
        self.save_memory = save_memory
        self.device = device
        self.stockcode2label = {}
        self.sector2label = {'n/a': -1}
        if market_name == 'NASDAQ' or market_name == 'NYSE':
            self.valid_start_index = 756
            self.test_start_index = 1008
        elif market_name == 'NASDAQ2':
            self.valid_start_index = 757
            self.test_start_index = 1008
        elif market_name == 'topix100':
            self.valid_start_index = 690
            self.test_start_index = 938
        elif market_name == 'ftse100':
            self.valid_start_index = 759
            self.test_start_index = 1012
        else:
            print('Invalid market')
            return
        print('Making pretrain data')
        train_x, train_y, valid_x, valid_y, test_x, test_y, train_mask, valid_mask, test_mask \
            = self.make_pretrain_dataset(market_name, seq_len, feature_describe)
        if self.save_memory:
            device = 'cpu'
        self.train_x = torch.Tensor(train_x).to(device)
        self.train_y = torch.Tensor(train_y).to(device)
        self.valid_x = torch.Tensor(valid_x).to(device)
        self.valid_y = torch.Tensor(valid_y).to(device)
        self.test_x = torch.Tensor(test_x).to(device)
        self.test_y = torch.Tensor(test_y).to(device)
        self.train_mask = torch.Tensor(train_mask).to(device)
        self.valid_mask = torch.Tensor(valid_mask).to(device)
        self.test_mask = torch.Tensor(test_mask).to(device)

        self.batch_x = ''
        self.batch_y = ''
        self.batch_masked_x = ''

    def get_batch_data(self, all_data, bs):
        return_data = []
        for i in range(all_data.shape[0] // bs + 1):
            if self.save_memory:
                return_data.append(all_data[i*bs:(i+1)*bs].to(self.device))
            else:
                return_data.append(all_data[i*bs:(i+1)*bs])
        return return_data

    def get_train_data(self, bs=1024, shuffle=False):
        self.batch_x, self.batch_y, self.batch_masked_x = '', '', ''
        if not shuffle:
            self.batch_x = self.get_batch_data(self.train_x, bs)
            self.batch_y = self.get_batch_data(self.train_y, bs)
            self.batch_masked_x = self.get_batch_data(self.train_mask, bs)
        else:
            rand_id = list(range(len(self.train_x)))
            random.shuffle(rand_id)
            self.batch_x = self.get_batch_data(self.train_x[rand_id], bs)
            self.batch_y = self.get_batch_data(self.train_y[rand_id], bs)
            self.batch_masked_x = self.get_batch_data(self.train_mask[rand_id], bs)

    def get_valid_data(self, bs=1024, shuffle=False):
        self.batch_x, self.batch_y, self.batch_masked_x = '', '', ''
        if not shuffle:
            self.batch_x = self.get_batch_data(self.valid_x, bs)
            self.batch_y = self.get_batch_data(self.valid_y, bs)
            self.batch_masked_x = self.get_batch_data(self.valid_mask, bs)
        else:
            rand_id = list(range(len(self.valid_x)))
            random.shuffle(rand_id)
            self.batch_x = self.get_batch_data(self.valid_x[rand_id], bs)
            self.batch_y = self.get_batch_data(self.valid_y[rand_id], bs)
            self.batch_masked_x = self.get_batch_data(self.valid_mask[rand_id], bs)

    def get_test_data(self, bs=1024, shuffle=False):
        self.batch_x, self.batch_y, self.batch_masked_x = '', '', ''
        if not shuffle:
            self.batch_x = self.get_batch_data(self.test_x, bs)
            self.batch_y = self.get_batch_data(self.test_y, bs)
            self.batch_masked_x = self.get_batch_data(self.test_mask, bs)
        else:
            rand_id = list(range(len(self.test_x)))
            random.shuffle(rand_id)
            self.batch_x = self.get_batch_data(self.test_x[rand_id], bs)
            self.batch_y = self.get_batch_data(self.test_y[rand_id], bs)
            self.batch_masked_x = self.get_batch_data(self.test_mask[rand_id], bs)

    def get_train_valid_data(self, bs=1024, shuffle=False):
        self.batch_x, self.batch_y, self.batch_masked_x = '', '', ''
        train_valid_x = torch.cat((self.train_x, self.valid_x), 0)
        train_valid_y = torch.cat((self.train_y, self.valid_y), 0)
        train_valid_masked_x = torch.cat((self.train_mask, self.valid_mask), 0)
        if not shuffle:
            self.batch_x = self.get_batch_data(train_valid_x, bs)
            self.batch_y = self.get_batch_data(train_valid_y, bs)
            self.batch_masked_x = self.get_batch_data(train_valid_masked_x, bs)
        else:
            rand_id = list(range(len(train_valid_x)))
            random.shuffle(rand_id)
            self.batch_x = self.get_batch_data(train_valid_x[rand_id], bs)
            self.batch_y = self.get_batch_data(train_valid_y[rand_id], bs)
            self.batch_masked_x = self.get_batch_data(train_valid_masked_x[rand_id], bs)

    def make_pretrain_dataset(self, market_name, seq_len, feature_describe):
        data_path = os.path.join(os.getcwd(), './data')
        with open(os.path.join(data_path, market_name + '_all_features.pkl'), 'rb') as fr:
            data = pkl.load(fr)
        all_stock_features = data['all_features']
        with open(os.path.join(data_path, './data',
                               market_name + '_sector_ticker.json'), 'r') as fr:
            sector_data = json.load(fr)

        stocklabel = 0
        for k in all_stock_features.keys():
            self.stockcode2label[k] = stocklabel
            stocklabel += 1
        sectorlabel = 0
        for k in sector_data.keys():
            if k == 'n/a':
                continue
            self.sector2label[k] = sectorlabel
            sectorlabel += 1

        train_x, train_y, valid_x, valid_y, test_x, test_y = [], [], [], [], [], []
        valid_start_index = self.valid_start_index
        test_start_index = self.test_start_index
        if feature_describe == 'all':
            feature_dim = list(range(25))
        elif feature_describe == 'close_only':
            feature_dim = [15, 16, 17, 18, 19]
        else:
            feature_dim = feature_describe
        close_dim = feature_dim.index(19)
        all_features = [all_stock_features[stock][:-1, 1:] for stock in all_stock_features.keys()]
        all_features = np.stack(all_features)
        labels = np.tile(np.arange(len(all_features)), (2, 1)).T
        t = []
        for k in sector_data.keys():
            t += sector_data[k]
            for stockcode in sector_data[k]:
                labels[self.stockcode2label[stockcode], 1] = self.sector2label[k]
        for i in range(seq_len-1, all_features.shape[1]):
            if i < valid_start_index:
                train_x.append(all_features[:, i-seq_len+1: i+1, feature_dim])
                train_y.append(labels)
            elif i < test_start_index:
                valid_x.append(all_features[:, i-seq_len+1: i+1, feature_dim])
                valid_y.append(labels)
            else:
                test_x.append(all_features[:, i-seq_len+1: i+1, feature_dim])
                test_y.append(labels)
        train_x = np.concatenate(train_x)
        train_y = np.concatenate(train_y)
        valid_x = np.concatenate(valid_x)
        valid_y = np.concatenate(valid_y)
        test_x = np.concatenate(test_x)
        test_y = np.concatenate(test_y)
        train_filter = np.any(train_x == -1234, axis=(1, 2))
        train_x = train_x[~train_filter]
        train_y = train_y[~train_filter]
        valid_filter = np.any(valid_x == -1234, axis=(1, 2))
        valid_x = valid_x[~valid_filter]
        valid_y = valid_y[~valid_filter]
        test_filter = np.any(test_x == -1234, axis=(1, 2))
        test_x = test_x[~test_filter]
        test_y = test_y[~test_filter]

        # add the third pretraining task, mask prices
        train_ave_y = train_x[:, :, close_dim].mean(1)
        valid_ave_y = valid_x[:, :, close_dim].mean(1)
        test_ave_y = test_x[:, :, close_dim].mean(1)
        train_y = np.column_stack((train_y, train_ave_y))
        valid_y = np.column_stack((valid_y, valid_ave_y))
        test_y = np.column_stack((test_y, test_ave_y))

        mask_num = round(train_x.shape[1] * abs(self.args.mask_rate))
        train_mask = np.ones(train_x.shape)
        valid_mask = np.ones(valid_x.shape)
        test_mask = np.ones(test_x.shape)
        mask_tensor = np.array([1] * (train_x.shape[1] - mask_num) + [0] * mask_num)
        for i in range(train_mask.shape[0]):
            for j in range(train_mask.shape[2]):
                np.random.shuffle(mask_tensor)
                train_mask[i, :, j] = mask_tensor
        for i in range(valid_mask.shape[0]):
            for j in range(valid_mask.shape[2]):
                np.random.shuffle(mask_tensor)
                valid_mask[i, :, j] = mask_tensor
        for i in range(test_mask.shape[0]):
            for j in range(test_mask.shape[2]):
                np.random.shuffle(mask_tensor)
                test_mask[i, :, j] = mask_tensor
        train_masked_x = train_x * train_mask
        valid_masked_x = valid_x * valid_mask
        test_masked_x = test_x * test_mask

        return train_x, train_y, valid_x, valid_y, test_x, test_y, train_masked_x, valid_masked_x, test_masked_x


if __name__ == "__main__":
    process_eod_data('NASDAQ')
    extract_sector_data('NASDAQ')