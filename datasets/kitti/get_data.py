"""Implements dataloaders for the KITTI dataset.

Here, the data is assumed to be in a folder titled "kitti".
"""
import numpy as np
from torch.utils.data import DataLoader



def get_dataloader(data_dir, batch_size=40, num_workers=8, 
                   train_shuffle=True, 
                   flatten_capacitance=False,
                   unsqueeze_channel=True, 
                   normalize_capacitance=True, 
                   normalize_depth=True,
                   task=0):
    """Get dataloaders for KITTI.

    Args:
        data_dir (str): Directory of data.
        batch_size (int, optional): Batch size. Defaults to 40.
        num_workers (int, optional): Number of workers. Defaults to 8.
        train_shuffle (bool, optional): Whether to shuffle training data or not. Defaults to True.
        flatten_imu (bool, optional): Whether to flatten imu data or not. Defaults to False.
        unsqueeze_channel (bool, optional): Whether to unsqueeze any channels or not. Defaults to True.
        normalize_imu (bool, optional): Whether to normalize the imus before returning. Defaults to True.
        normalize_audio (bool, optional): Whether to normalize the audio before returning. Defaults to True.

    Returns:
        tuple: Tuple of (training dataloader, validation dataloader, test dataloader)
    """
    trains = [np.load(data_dir+"/capacitance/train_data.npy"), np.load(data_dir +
                                                                 "/depth/train_data.npy"), np.load(data_dir +
                                                                 "/touch/train_data.npy"), np.load(data_dir+"/gesture/train_data.npy")]
    tests = [np.load(data_dir+"/capacitance/test_data.npy"), np.load(data_dir +
                                                               "/depth/test_data.npy"), np.load(data_dir +
                                                               "/touch/test_data.npy"), np.load(data_dir+"/gesture/test_data.npy")]
    print('trains:', trains[0].shape, trains[1].shape, trains[2].shape, trains[3].shape)
    # (58771, 41, 72) (58771, 41, 72) (58771,) (58771,)
    print('tests:', tests[0].shape, tests[1].shape, tests[2].shape, tests[3].shape)
    # (6603, 41, 72) (6603, 41, 72) (6603,) (6603,)
    # print unique on touch
    print(np.unique(trains[2]))
    print(np.unique(tests[3]))
    if flatten_capacitance:
        trains[0] = trains[0].reshape(58771, 41*72)
        tests[0] = tests[0].reshape(6603, 41*72)
    if normalize_capacitance:
        trains[0] /= 255.0
        tests[0] /= 255.0
    if normalize_depth:
        trains[1] = trains[1]/255.0
        tests[1] = tests[1]/255.0
    if unsqueeze_channel:
        trains[0] = np.expand_dims(trains[0], 1)
        tests[0] = np.expand_dims(tests[0], 1)
        trains[1] = np.expand_dims(trains[1], 1)
        tests[1] = np.expand_dims(tests[1], 1)
    trains[2] = trains[2].astype(int)
    tests[2] = tests[2].astype(int)
    if task == 0:
        trainlist = [[trains[j][i] for j in [0,1,2]] for i in range(58771)]
        testlist = [[tests[j][i] for j in [0,1,2]] for i in range(6603)]
    elif task == 1:
        trainlist = [[trains[j][i] for j in [0,1,3]] for i in range(58771)]
        testlist = [[tests[j][i] for j in [0,1,3]] for i in range(6603)]
    valids = DataLoader(trainlist[54070:58771], shuffle=False,
                        num_workers=num_workers, batch_size=batch_size)
    tests = DataLoader(testlist, shuffle=False,
                       num_workers=num_workers, batch_size=batch_size)
    trains = DataLoader(trainlist[0:54070], shuffle=train_shuffle,
                        num_workers=num_workers, batch_size=batch_size, drop_last=True)
    return trains, valids, tests

