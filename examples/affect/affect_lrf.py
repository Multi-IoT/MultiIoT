import torch
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))


from private_test_scripts_v0.all_in_one import all_in_one_train # noqa
from training_structures.Supervised_Learning import train, test # noqa
from unimodals.common_models import GRUWithLinear, MLP # noqa
from datasets.affect.get_data import get_dataloader # noqa
from fusions.common_fusions import Concat, LowRankTensorFusion # noqa


# mosi_data.pkl, mosei_senti_data.pkl
# mosi_raw.pkl, mosei_raw.pkl, sarcasm.pkl, humor.pkl
# raw_path: mosi.hdf5, mosei.hdf5, sarcasm_raw_text.pkl, humor_raw_text.pkl
traindata, validdata, test_robust = \
    get_dataloader('/home/paul/MultiBench/mosi_raw.pkl', robust_test=False)

# mosi/mosei
encoders = [GRUWithLinear(35, 64, 32, dropout=True, has_padding=True).cuda(),
            GRUWithLinear(74, 128, 32, dropout=True, has_padding=True).cuda(),
            GRUWithLinear(300, 512, 128, dropout=True, has_padding=True).cuda()]
head = MLP(128, 512, 1).cuda()

# humor/sarcasm
# encoders=[GRUWithLinear(371,512,32,dropout=True,has_padding=True).cuda(), \
#     GRUWithLinear(81,256,32,dropout=True,has_padding=True).cuda(),\
#     GRUWithLinear(300,600,128,dropout=True,has_padding=True).cuda()]
# head=MLP(128,512,1).cuda()

fusion = LowRankTensorFusion([32, 32, 128], 128, 32).cuda()

train(encoders, fusion, head, traindata, validdata, 100, task="regression", optimtype=torch.optim.AdamW,
      early_stop=True, is_packed=True, lr=1e-3, save='mosi_lf_best.pt', weight_decay=0.01, objective=torch.nn.L1Loss())

print("Testing:")
model = torch.load('mosi_lf_best.pt').cuda()

test(model=model, test_dataloaders_all=test_robust, dataset='mosi', is_packed=True,
     criterion=torch.nn.L1Loss(), task='posneg-classification', no_robust=True)
