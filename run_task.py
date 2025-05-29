import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_args, random_seed_control
from data import MyDataLoader, MyPretrainDataLoader
from model import TransformerStockPrediction
from metrics import return_eval
import os
import time


class Stock_Selection():
    def __init__(self, args, stocksl_dataloader, pretrain_dataloader, pretrain_inf, device):
        self.args = args
        self.device = device
        self.train_epochs = args.epoch
        self.pretrain_epochs = args.pretrain_epoch
        self.pretrain_bs = args.batch_size
        self.stocksl_dataloader = stocksl_dataloader
        self.pretrain_dataloader = pretrain_dataloader
        self.loss_reg = True
        self.loss_rank = True
        self.loss_alpha = args.loss_alpha
        self.topk = args.topk
        self.train_shuffle = args.shuffle
        self.train_train = args.train_train
        self.train_valid = args.train_valid
        if self.train_train:
            self.model_tt = TransformerStockPrediction(input_size=self.stocksl_dataloader.train_x.shape[3],
                num_class=1, hidden_size=args.hidden_size,
                num_feat_att_layers=args.num_feat_att_layers,
                num_pre_att_layers=args.num_pre_att_layers,
                num_heads=args.num_heads, days=args.days, dropout=args.dropout).to(device)
        if self.train_valid:
            self.model_tv = TransformerStockPrediction(input_size=self.stocksl_dataloader.train_x.shape[3],
                num_class=1, hidden_size=args.hidden_size,
                num_feat_att_layers=args.num_feat_att_layers,
                num_pre_att_layers=args.num_pre_att_layers,
                num_heads=args.num_heads, days=args.days, dropout=args.dropout).to(device)
        self.pretrain_tasks = args.pretrain_tasks
        for task_id, name in enumerate(self.pretrain_tasks):
            task_num_class = pretrain_inf[args.market_name][name]
            if name == 'mask_avg_price':
                if self.args.mask_rate < 0:
                    # ablation study, recover specific masked values
                    task_num_class = args.days
                else:
                    task_num_class = 1
            if self.train_train:
                self.model_tt.add_outlayer(name=name, num_class=task_num_class, device=self.device)
            if self.train_valid:
                self.model_tv.add_outlayer(name=name, num_class=task_num_class, device=self.device)
        if self.train_train:
            self.optimizer_tt = optim.Adam(self.model_tt.parameters(), lr=args.lr)
        if self.train_valid:
            self.optimizer_tv = optim.Adam(self.model_tv.parameters(), lr=args.lr)
        self.classification_loss = nn.CrossEntropyLoss()
        pretrain_loss_coef = args.pretrain_coef.split('-')
        self.pretrain_loss_coef = {'stock': float(pretrain_loss_coef[0]),
                                   'sector': float(pretrain_loss_coef[1]),
                                   'mask_avg_price': float(pretrain_loss_coef[2])}

    def loss(self, pred, train_label, train_mask):
        base_price = train_label[:, 0:1]
        obj_price = train_label[:, 1:2]
        pred_ratio = (pred - base_price) / base_price
        obj_ratio = (obj_price - base_price) / base_price

        losses = {}
        all_loss = 0.0
        if self.loss_reg:
            reg_loss = (train_mask * ((pred_ratio - obj_ratio) ** 2).squeeze()).sum() / train_mask.sum()
            losses['reg_loss'] = reg_loss
            all_loss += reg_loss
        if self.loss_rank:
            all_ones = torch.ones(len(pred), 1).to(self.device)
            pre_pw_dif = (torch.matmul(pred_ratio, torch.transpose(all_ones, 0, 1))
                          - torch.matmul(all_ones, torch.transpose(pred_ratio, 0, 1)))
            gt_pw_dif = (torch.matmul(all_ones, torch.transpose(pred_ratio, 0, 1))
                         - torch.matmul(obj_ratio, torch.transpose(all_ones, 0, 1)))
            mask_pw = torch.matmul(train_mask.unsqueeze(1), train_mask.unsqueeze(0))
            rank_loss = (nn.functional.relu(pre_pw_dif * gt_pw_dif * mask_pw)).sum() / mask_pw.sum()
            losses['rank_loss'] = rank_loss
            all_loss += self.loss_alpha * rank_loss
        losses['train_loss'] = all_loss
        return losses

    def train_epoch(self, train_data, train_label, train_mask, model, optimizer):
        model.train()
        if self.loss_reg:
            reg_loss = 0.0
        if self.loss_rank:
            rank_loss = 0.0
        train_loss = 0.0
        for i in range(len(train_data)):
            outputs = model(train_data[i])
            optimizer.zero_grad()
            losses = self.loss(outputs, train_label[i], train_mask[i])
            loss = losses['train_loss']
            loss.backward()
            optimizer.step()
            if self.loss_reg:
                reg_loss += losses['reg_loss'].item()
            if self.loss_rank:
                rank_loss += losses['rank_loss'].item()
            train_loss += loss.item()
        train_loss = train_loss / len(train_data)
        print('Train Loss: {:.7f}'.format(train_loss), end='   ')
        if self.loss_reg:
            reg_loss = reg_loss / len(train_data)
            print('Regression Loss: {:.7f}'.format(reg_loss), end='   ')
        if self.loss_rank:
            rank_loss = rank_loss / len(train_data)
            print('Ranking Loss: {:.7f}'.format(rank_loss), end='   ')
        print('')
        return train_loss

    def valid(self, valid_data, valid_label, valid_mask, model):
        model.eval()
        all_outputs = []
        if self.loss_reg:
            reg_loss = 0.0
        if self.loss_rank:
            rank_loss = 0.0
        valid_loss = 0.0
        with torch.no_grad():
            for i in range(len(valid_data)):
                outputs = model(valid_data[i])
                all_outputs.append(outputs)
                losses = self.loss(outputs, valid_label[i], valid_mask[i])
                if self.loss_reg:
                    reg_loss += losses['reg_loss'].item()
                if self.loss_rank:
                    rank_loss += losses['rank_loss'].item()
                valid_loss += losses['train_loss'].item()
        all_outputs = torch.cat(all_outputs, 1).transpose(0, 1)
        pnls, srs = return_eval(all_outputs, valid_label, valid_mask, self.topk)

        valid_loss = valid_loss / len(valid_data)
        print('Valid Loss: {:.7f}'.format(valid_loss), end='   ')
        if self.loss_reg:
            reg_loss = reg_loss / len(valid_data)
            print('Regression Loss: {:.7f}'.format(reg_loss), end='   ')
        if self.loss_rank:
            rank_loss = rank_loss / len(valid_data)
            print('Ranking Loss: {:.7f}'.format(rank_loss), end='   ')
        print('')
        print('Valid', end='   ')
        for topk_id in range(len(self.topk)):
            print('|Top', self.topk[topk_id], end='|: ')
            print('PnL: {:.4f}'.format(pnls[topk_id]), end=' ')
            print('SR: {:.4f}'.format(srs[topk_id]), end='   ')
        print('')
        return pnls, srs

    def test(self, test_data, test_label, test_mask, model):
        model.eval()
        all_outputs = []
        with torch.no_grad():
            for i in range(len(test_data)):
                outputs = model(test_data[i])
                all_outputs.append(outputs)
        all_outputs = torch.cat(all_outputs, 1).transpose(0, 1)
        pnls, srs = return_eval(all_outputs, test_label, test_mask, self.topk)
        print('Test', end='   ')
        for topk_id in range(len(self.topk)):
            print('|Top', self.topk[topk_id], end='|: ')
            print('PnL: {:.4f}'.format(pnls[topk_id]), end=' ')
            print('SR: {:.4f}'.format(srs[topk_id]), end='   ')
        print('')
        return pnls, srs

    def get_pretrain_output(self, data, masked_data, model, tasks):
        all_outputs = {}
        for task in tasks:
            model.pretrain_task = task
            if task == 'mask_avg_price':
                outputs = model(masked_data)
            else:
                outputs = model(data)
            all_outputs[task] = outputs
        return all_outputs

    def pretrain_train_epoch(self, train_data, train_label, train_masked_x, model, optimizer, tasks):
        model.train()
        for i, train_batch_x in enumerate(train_data):
            all_outputs = self.get_pretrain_output(train_batch_x, train_masked_x[i], model, tasks)
            all_loss = 0
            all_loss_record = {}
            optimizer.zero_grad()
            for task in tasks:
                temp_label = train_label[i][:, self.pretrain_tasks.index(task)]
                if task == 'mask_avg_price':
                    if self.args.mask_rate < 0:
                        # only for 'all' features
                        loss = ((all_outputs[task].squeeze() - train_batch_x[:, :, 19]) ** 2).sum() / (train_masked_x[i][:, :, 19] == 0).sum()
                    else:
                        loss = ((all_outputs[task].squeeze() - temp_label)**2).mean()
                else:
                    mask = (temp_label >= 0)
                    loss = self.classification_loss(all_outputs[task][mask], temp_label[mask].long())
                all_loss += self.pretrain_loss_coef[task] * loss
                all_loss_record[task] = loss
            all_loss.backward()
            optimizer.step()
            if (i+1) % 200 == 0:
                for task in tasks:
                    print("Batch {}/{} Loss {}: {:.7f}".format(i, len(train_data), task, all_loss_record[task]))
        return loss

    def pretrain_valid(self, valid_data, valid_label, valid_masked_x, model, tasks):
        model.eval()
        results = {}
        for task in tasks:
            results[task] = 0
        data_num = 0
        for i, valid_batch_x in enumerate(valid_data):
            all_outputs = self.get_pretrain_output(valid_batch_x, valid_masked_x[i], model, tasks)
            for task in tasks:
                temp_label = valid_label[i][:, self.pretrain_tasks.index(task)]
                mask = (temp_label >= 0)
                if task == 'mask_avg_price':
                    if self.args.mask_rate < 0:
                        # only for 'all' features
                        results[task] += ((all_outputs[task].squeeze() - valid_batch_x[:, :, 19]) ** 2).sum().item()
                        data_num += (valid_masked_x[i][:, :, 19] == 0).sum().item()
                    else:
                        results[task] += ((all_outputs[task].squeeze() - temp_label)**2).sum().item()
                        data_num += len(temp_label)
                else:
                    results[task] += (torch.argmax(all_outputs[task][mask], 1) == temp_label[mask]).sum().item()
                    data_num += mask.sum().item()
        for task in tasks:
            results[task] = results[task] / data_num
            print('Validation Result {} {:.7f}'.format(task, results[task]))
        return results

    def pretrain_test(self, test_data, test_label, test_masked_x, model, tasks):
        model.eval()
        results = {}
        for task in tasks:
            results[task] = 0
        data_num = 0
        for i, test_batch_x in enumerate(test_data):
            all_outputs = self.get_pretrain_output(test_batch_x, test_masked_x[i], model, tasks)
            for task in tasks:
                temp_label = test_label[i][:, self.pretrain_tasks.index(task)]
                mask = (temp_label >= 0)
                if task == 'mask_avg_price':
                    if self.args.mask_rate < 0:
                        # only for 'all' features
                        results[task] += ((all_outputs[task].squeeze() - test_batch_x[:, :, 19]) ** 2).sum().item()
                        data_num += (test_masked_x[i][:, :, 19] == 0).sum().item()
                    else:
                        results[task] += ((all_outputs[task].squeeze() - temp_label)**2).sum().item()
                        data_num += len(temp_label)
                else:
                    results[task] += (torch.argmax(all_outputs[task][mask], 1) == temp_label[mask]).sum().item()
                    data_num += mask.sum().item()
        for task in tasks:
            results[task] = results[task] / data_num
            print('Test Result {} {:.7f}'.format(task, results[task]))
        return results

    def run_pretrain(self, task):
        if len(task) == 0:
            return
        for epoch in range(self.pretrain_epochs):
            print('-------------Epoch', epoch, '----------------')
            if self.train_train:
                print('Train-Train')
                self.pretrain_dataloader.get_train_data(self.pretrain_bs, True)
                self.pretrain_train_epoch(self.pretrain_dataloader.batch_x, self.pretrain_dataloader.batch_y,
                                          self.pretrain_dataloader.batch_masked_x, self.model_tt, self.optimizer_tt, task)
                self.pretrain_dataloader.get_valid_data(self.pretrain_bs)
                self.pretrain_valid(self.pretrain_dataloader.batch_x, self.pretrain_dataloader.batch_y,
                                    self.pretrain_dataloader.batch_masked_x, self.model_tt, task)
            if self.train_valid:
                print('Train-Valid')
                self.pretrain_dataloader.get_train_valid_data(self.pretrain_bs, True)
                self.pretrain_train_epoch(self.pretrain_dataloader.batch_x, self.pretrain_dataloader.batch_y,
                                          self.pretrain_dataloader.batch_masked_x, self.model_tv, self.optimizer_tv, task)
            if epoch + 1 == self.pretrain_epochs:
                self.save_models(epoch=epoch + 1)

        self.pretrain_dataloader.get_test_data(self.pretrain_bs)
        self.pretrain_test(self.pretrain_dataloader.batch_x, self.pretrain_dataloader.batch_y,
                           self.pretrain_dataloader.batch_masked_x, self.model_tt, task)


    def run_stocksl_exp(self, finetune=False):
        if self.train_train:
            self.model_tt.pretrain_task = ''
        if self.train_valid:
            self.model_tv.pretrain_task = ''
        if finetune:
            if self.train_train:
                self.model_tt.change_finetune_mode(True, freezing=self.args.freezing)
            if self.train_valid:
                self.model_tv.change_finetune_mode(True, freezing=self.args.freezing)
        for epoch in range(self.train_epochs):
            print('-------------Epoch', epoch, '----------------')
            if self.train_train:
                print('Train-Train')
                self.stocksl_dataloader.get_train_data()
                self.train_epoch(self.stocksl_dataloader.batch_x, self.stocksl_dataloader.batch_y,
                                 self.stocksl_dataloader.batch_mask, self.model_tt, self.optimizer_tt)
                self.stocksl_dataloader.get_valid_data()
                self.valid(self.stocksl_dataloader.batch_x, self.stocksl_dataloader.batch_y,
                           self.stocksl_dataloader.batch_mask, self.model_tt)
                self.stocksl_dataloader.get_test_data()
                self.test(self.stocksl_dataloader.batch_x, self.stocksl_dataloader.batch_y,
                          self.stocksl_dataloader.batch_mask, self.model_tt)
            if self.train_valid:
                print('Train-Valid')
                self.stocksl_dataloader.get_train_valid_data(self.train_shuffle)
                self.train_epoch(self.stocksl_dataloader.batch_x, self.stocksl_dataloader.batch_y,
                                 self.stocksl_dataloader.batch_mask, self.model_tv, self.optimizer_tv)
                self.stocksl_dataloader.get_test_data()
                self.test(self.stocksl_dataloader.batch_x, self.stocksl_dataloader.batch_y,
                          self.stocksl_dataloader.batch_mask, self.model_tv)
        if finetune:
            if self.train_train:
                self.model_tt.change_finetune_mode(False)
            if self.train_valid:
                self.model_tv.change_finetune_mode(False)

    def save_models(self, epoch, path=''):
        if path == '':
            path = "./models/pre_train_models/"\
                   f"market-{args.market_name}_days-{args.days}_feature-describe-{args.feature_describe}" \
                   f"_ongoing-task-{'-'.join(args.ongoing_task)}_mask_rate-{args.mask_rate}_lr-{args.lr}_pretrain-coefs-{args.pretrain_coef}"
        os.makedirs(path, exist_ok=True)
        print("Saving models to {}".format(path))
        if self.train_train:
            self.model_tt.save_model(os.path.join(path, 'model_tt_' + str(epoch) + '.ckpt'''))
        if self.train_valid:
            self.model_tv.save_model(os.path.join(path, 'model_tv_' + str(epoch) + '.ckpt'''))

    def load_models(self, epoch, path=''):
        if path == '':
            path = "./models/pre_train_models/"\
                   f"market-{args.market_name}_days-{args.days}_feature-describe-{args.feature_describe}" \
                   f"_ongoing-task-{'-'.join(args.ongoing_task)}_lr-{args.lr}"
        print("Loading models from {}".format(path))
        if self.train_train:
            self.model_tt.load_model(os.path.join(path, 'model_tt_' + str(epoch) + '.ckpt'''))
        if self.train_valid:
            self.model_tv.load_model(os.path.join(path, 'model_tv_' + str(epoch) + '.ckpt'''))


if __name__ == '__main__':
    start_time = time.time()

    args = get_args()
    random_seed_control(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pre_train_task = args.ongoing_task
    if len(pre_train_task) == 0:
        finetune = False
    else:
        finetune = True
    print(args)
    print(finetune)

    pretrain_inf = {'NASDAQ': {'stock': 1026, 'sector': 112, 'mask_avg_price': 1},
                    'NYSE': {'stock': 1737, 'sector': 129, 'mask_avg_price': 1},
                    'topix100': {'stock': 95, 'sector': 10, 'mask_avg_price': 1},
                    'NASDAQ2': {'stock': 718, 'sector': 106, 'mask_avg_price': 1},
                    'ftse100': {'stock': 87, 'sector': 11, 'mask_avg_price': 1},
                    }
    stocksl_dataloader = MyDataLoader(args=args, market_name=args.market_name, seq_len=args.days,
                                      feature_describe=args.feature_describe, save_memory=args.save_memory,
                                      device='cuda')
    if args.pretrain_epoch > 0:
        pretrain_dataloader = MyPretrainDataLoader(args=args, market_name=args.market_name, seq_len=args.days,
                                                   feature_describe=args.feature_describe, save_memory=args.save_memory,
                                                   device='cuda')
    else:
        pretrain_dataloader = 'unnecessary'
    print(time.time() - start_time)

    task_StockSelection = Stock_Selection(args, stocksl_dataloader, pretrain_dataloader, pretrain_inf, device)
    if args.pretrain_epoch > 0:
        task_StockSelection.run_pretrain(pre_train_task)
        if args.save_pretrain == 1:
            task_StockSelection.save_models(epoch=args.pretrain_epoch)
    if args.epoch > 0:
        if args.load_path != '':
            task_StockSelection.load_models(epoch=args.epoch, path=args.load_path)
        task_StockSelection.run_stocksl_exp(finetune=finetune)

    end_time = time.time()
    running_time = end_time - start_time
    print('Time (h):', running_time / 3600, 'Time (min):', running_time / 60)
