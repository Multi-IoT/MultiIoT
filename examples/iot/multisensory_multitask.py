import sys
import os
sys.path.insert(1,os.getcwd())
#from perceiver_pytorch.multi_modality_perceiver import MultiModalityPerceiver, InputModality
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from datasets.samosa.get_data import get_dataloader
trains1,valid1,test1=get_dataloader('/path/to/MultiIoT', task=0)
from datasets.samosa.get_data import get_dataloader
trains2,valid2,test2=get_dataloader('/path/to/MultiIoT', task=1)
from datasets.touchpose.get_data import get_dataloader
trains3,valid3,test3=get_dataloader('/path/to/MultiIoT', task=0)
from datasets.touchpose.get_data import get_dataloader
trains4,valid4,test4=get_dataloader('/path/to/MultiIoT', task=1)

from perceivers.crossattnperceiver import MultiModalityPerceiver, InputModality
device='cuda:0'
imu_modality=InputModality(
    name='imu',
    input_channels=100,
    input_axis=2,
    num_freq_bands=6,
    max_freq=1
)
audio_modality=InputModality(
    name='audio',
    input_channels=100,
    input_axis=1,
    num_freq_bands=6,
    max_freq=1
)
capacitance_modality=InputModality(
    name='capacitance',
    input_channels=41,
    input_axis=2,
    num_freq_bands=6,
    max_freq=1
)
depth_modality=InputModality(
    name='depth',
    input_channels=41,
    input_axis=2,
    num_freq_bands=6,
    max_freq=1
)

feature1_modality=InputModality(
    name='feature1',
    input_channels=35,
    input_axis=1,
    num_freq_bands=3,
    max_freq=1
)
feature2_modality=InputModality(
    name='feature2',
    input_channels=74,
    input_axis=1,
    num_freq_bands=3,
    max_freq=1
)
feature3_modality=InputModality(
    name='feature3',
    input_channels=300,
    input_axis=1,
    num_freq_bands=3,
    max_freq=1
)
feature4_modality=InputModality(
    name='feature4',
    input_channels=371,
    input_axis=1,
    num_freq_bands=3,
    max_freq=1
)
feature5_modality=InputModality(
    name='feature5',
    input_channels=81,
    input_axis=1,
    num_freq_bands=3,
    max_freq=1
)
for i in range(1):
    #"""
    model = MultiModalityPerceiver(
        modalities=(imu_modality,audio_modality,capacitance_modality,depth_modality,feature1_modality,feature2_modality,feature3_modality,feature4_modality,feature5_modality),
        depth=1,  # depth of net, combined with num_latent_blocks_per_layer to produce full Perceiver
        num_latents=20,
        # number of latents, or induced set points, or centroids. different papers giving it different names
        latent_dim=64,  # latent dimension
        cross_heads=1,  # number of heads for cross attention. paper said 1
        latent_heads=8,  # number of heads for latent self attention, 8
        cross_dim_head=64,
        latent_dim_head=64,
        num_classes=1,  # output number of classes
        attn_dropout=0.,
        ff_dropout=0.,
        #embed=True,
        weight_tie_layers=True,
        num_latent_blocks_per_layer=1, # Note that this parameter is 1 in the original Lucidrain implementation,
        cross_depth=1
    ).to(device)
    model.to_logitslist=torch.nn.ModuleList([torch.nn.Sequential(torch.nn.LayerNorm(128),torch.nn.Linear(128,27)),torch.nn.Sequential(torch.nn.LayerNorm(128),torch.nn.Linear(128,6)),torch.nn.Sequential(torch.nn.LayerNorm(384),torch.nn.Linear(384,12)),torch.nn.Sequential(torch.nn.LayerNorm(384),torch.nn.Linear(384,15))]).to(device)

    from perceivers.train_structure_multitask import train

    records=train(model,80,[trains1,trains2,trains3,trains4],[valid1,valid2,valid3,valid4],[test1,test2,test3,test4],[['imu','audio'],['imu','audio'],['capacitance','depth'],['capacitance','depth']],'./multitask.pt',lr=0.0008,device=device,train_weights=[1.2,0.9,1.1,1.0],is_affect=[False,False,False,False],unsqueezing=[True,True,False,False],transpose=[False,False,False,False],evalweights=[1,1,1,1],start_from=0,weight_decay=0.001)
