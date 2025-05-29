import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_args, random_seed_control
from data_simulate import MySimuldateDataLoader
from model import TransformerStockPrediction
import time


class Simulate_Task():
    def __init__(self, args, simulate_dataloader, device):
        self.simulate_dataloader = simulate_dataloader
        self.device = device
        self.pretrain_epochs = 100
        self.train_epochs =100
        # self.model = TransformerStockPrediction(input_size=self.simulate_dataloader.train_x.shape[-1],
        #                                         num_class=1, hidden_size=args.hidden_size,
        #                                         num_feat_att_layers=args.num_feat_att_layers,
        #                                         num_pre_att_layers=args.num_pre_att_layers,
        #                                         num_heads=args.num_heads, days=32, dropout=args.dropout).to(device)
        self.model = TransformerStockPrediction(input_size=self.simulate_dataloader.train_x.shape[-1],
                                                num_class=1, hidden_size=32,
                                                num_feat_att_layers=args.num_feat_att_layers,
                                                num_pre_att_layers=args.num_pre_att_layers,
                                                num_heads=args.num_heads, days=32, dropout=args.dropout).to(device)
        self.model.add_outlayer(name='stock', num_class=10, device=self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.batch_size = 4096
        self.classification_loss = nn.CrossEntropyLoss()

    def train_epoch(self, train_data, train_label, model, optimizer):
        model.train()
        for i in range(len(train_data) // self.batch_size + 1):
            train_batch_x = train_data[i * self.batch_size:(i + 1) * self.batch_size]
            train_batch_y = train_label[i * self.batch_size:(i + 1) * self.batch_size][:, 1]
            outputs = model(train_batch_x)
            loss = ((outputs.squeeze() - train_batch_y) ** 2).mean()
            loss.backward()
            optimizer.step()
        print('Train Loss: {:.7f}'.format(loss), end='   ')
        return 0

    def test(self, test_data, test_label, model):
        model.eval()
        all_outputs = []
        with torch.no_grad():
            for i in range(len(test_data) // self.batch_size + 1):
                test_batch_x = test_data[i * self.batch_size:(i + 1) * self.batch_size]
                outputs = model(test_batch_x)
                all_outputs.append(outputs.squeeze())
        all_outputs = torch.cat(all_outputs, 0)
        test_y = test_label[:, 1]
        mse = ((all_outputs - test_y) ** 2).mean()
        print('Test mse', mse)
        return mse

    def pretrain_train_epoch(self, train_data, train_label, model, optimizer, task):
        model.train()
        model.pretrain_task = task
        for i in range(len(train_data) // self.batch_size + 1):
            train_batch_x = train_data[i * self.batch_size:(i + 1) * self.batch_size]
            train_batch_y = train_label[i * self.batch_size:(i + 1) * self.batch_size][:, 0].long()
            outputs = model(train_batch_x)
            loss = self.classification_loss(outputs, train_batch_y)
            loss.backward()
            optimizer.step()
            # if (i+1) % 20 == 0:
            #     print("Batch {}/{} Loss {:.7f}".format(i, (len(train_data) // self.batch_size + 1), loss))
        print('Pre-train Loss: {:.7f}'.format(loss), end='   ')
        return 0

    def pretrain_test_epoch(self, train_data, train_label, model, optimizer, task):
        model.eval()
        model.pretrain_task = task
        all_outputs = []
        for i in range(len(train_data) // self.batch_size + 1):
            train_batch_x = train_data[i * self.batch_size:(i + 1) * self.batch_size]
            outputs = model(train_batch_x)
            all_outputs.append(outputs)
        all_outputs = torch.cat(all_outputs, 0)
        train_batch_y = train_label[:, 0].long()
        acc = (torch.argmax(all_outputs, 1) == train_batch_y).sum().item() / len(train_data)
        print('Test acc', acc)
        return acc

    def run_pretrain(self, task='stock'):
        max_valid_acc = 0
        selected_test_acc = 0
        for epoch in range(self.pretrain_epochs):
            print('Epoch {}/{}'.format(epoch, self.pretrain_epochs))
            self.pretrain_train_epoch(self.simulate_dataloader.train_valid_x, self.simulate_dataloader.train_valid_y, self.model,
                                      self.optimizer, task)
            valid_acc = self.pretrain_test_epoch(self.simulate_dataloader.valid_x, self.simulate_dataloader.valid_y, self.model,
                                           self.optimizer, task)
            test_acc = self.pretrain_test_epoch(self.simulate_dataloader.test_x, self.simulate_dataloader.test_y, self.model,
                                      self.optimizer, task)
            if valid_acc > max_valid_acc:
                max_valid_acc = valid_acc
                selected_test_acc = test_acc
        return selected_test_acc

    def run_predict(self, finetune=False):
        self.model.pretrain_task = ''
        if finetune:
            self.model.change_finetune_mode(True)
        min_valid_mse = 100
        selected_test_mse = 0
        for epoch in range(self.train_epochs):
            print('Epoch {}/{}'.format(epoch, self.train_epochs))
            self.train_epoch(self.simulate_dataloader.train_x, self.simulate_dataloader.train_y, self.model, self.optimizer)
            valid_mse = self.test(self.simulate_dataloader.valid_x, self.simulate_dataloader.valid_y, self.model)
            test_mse = self.test(self.simulate_dataloader.test_x, self.simulate_dataloader.test_y, self.model)
            if valid_mse < min_valid_mse:
                min_valid_mse = valid_mse
                selected_test_mse = test_mse
        if finetune:
            self.model.change_finetune_mode(False)
        return selected_test_mse



if __name__ == '__main__':
    start_time = time.time()
    args = get_args()
    random_seed_control(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simulate_dataloader = MySimuldateDataLoader(device)
    simulate_task_runner = Simulate_Task(args, simulate_dataloader, device)

    finetune_mode = False
    if args.pretrain_epoch != 0:
        pre_train_acc = simulate_task_runner.run_pretrain()
        finetune_mode = True
    # mse = simulate_task_runner.run_predict(finetune_mode)
    if args.pretrain_epoch != 0:
        print("Pretrain Acc: ", pre_train_acc)
    # print("Test MSE: ", mse)
    # pdb.set_trace()

