import argparse
import random
import numpy as np
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--market_name', type=str, default='NASDAQ')
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--save_memory', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--pretrain_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_feat_att_layers', type=int, default=1)
    parser.add_argument('--num_pre_att_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--extra_info', type=str, default='')
    parser.add_argument('--get_features', type=str, default='')
    parser.add_argument('--loss_alpha', type=int, default=5)
    parser.add_argument('--topk', type=int, nargs='+', default=[1, 5, 10])
    parser.add_argument('--days', type=int, default=16)
    parser.add_argument('--feature_describe', type=str, default='close_only')
    parser.add_argument('--shuffle', type=int, default=1)
    parser.add_argument('--train_train', type=int, default=1)
    parser.add_argument('--train_valid', type=int, default=1)
    parser.add_argument('--validby', type=str, default='acc')
    parser.add_argument('--pretrain_tasks', type=str, nargs='+', default=['stock', 'sector', 'mask_avg_price'])
    parser.add_argument('--ongoing_task', type=str, nargs='+', default=[])
    parser.add_argument('--mask_rate', type=float, default=0.3)
    parser.add_argument('--pretrain_coef', type=str, default='1-1-1')
    parser.add_argument('--freezing', type=str, default='embedding')
    parser.add_argument('--save_pretrain', type=int, default=0)
    parser.add_argument('--load_path', type=str, default='')
    return parser.parse_args()

def random_seed_control(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
