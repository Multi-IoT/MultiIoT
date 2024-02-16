import sys
import os
sys.path.append(os.getcwd())

from unimodals.common_models import LeNet, MLP, Constant
import torch
from torch import nn
from datasets.samosa.get_data import get_dataloader
from training_structures.unimodal import train, test

modalnum = 0
traindata, validdata, testdata = get_dataloader(
    '../../../data/MultiIoT/SAMoSA')
channels = 3
# encoders=[LeNet(1,channels,3).cuda(),LeNet(1,channels,5).cuda()]
encoder = LeNet(1, channels, 2).cuda()
head = MLP(channels*48, 40, 27).cuda()


train(encoder, head, traindata, validdata, 100, optimtype=torch.optim.SGD,
      lr=0.01, weight_decay=0.0001, modalnum=modalnum)

print("Testing:")
encoder = torch.load('encoder.pt').cuda()
head = torch.load('head.pt')
test(encoder, head, testdata, modalnum=modalnum, no_robust=True)
