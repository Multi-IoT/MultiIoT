from torch import nn
import torch.nn.functional as F
import torch
import pmdarima
import numpy as np
import argparse
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

sys.path.append('/home/pliang/multibench/MultiBench/datasets/stocks')


from private_test_scripts_v0.all_in_one import all_in_one_train, all_in_one_test # noqa
from training_structures.Supervised_Learning import train, test # noqa
from datasets.stocks.get_data import get_dataloader # noqa
from unimodals.common_models import LSTM, Identity # noqa
from fusions.common_fusions import Stack # noqa


parser = argparse.ArgumentParser()
parser.add_argument('--input-stocks', metavar='input', help='input stocks')
parser.add_argument('--target-stock', metavar='target', help='target stock')
args = parser.parse_args()
print('Input: ' + args.input_stocks)
print('Target: ' + args.target_stock)


stocks = sorted(args.input_stocks.split(' '))
train_loader, val_loader, test_loader = get_dataloader(
    stocks, stocks, [args.target_stock])

n_modalities = len(train_loader.dataset[0]) - 1
encoders = [Identity().cuda()] * n_modalities
fusion = Stack().cuda()
head = LSTM(n_modalities, 128, linear_layer_outdim=1).cuda()
allmodules = [*encoders, fusion, head]


def trainprocess():
    train(encoders, fusion, head, train_loader, val_loader, total_epochs=2,
          task='regression', optimtype=torch.optim.Adam, objective=nn.MSELoss())


all_in_one_train(trainprocess, allmodules)

model = torch.load('best.pt').cuda()
test(model, test_loader, dataset='finance F&B',
     task='regression', criterion=nn.MSELoss())
