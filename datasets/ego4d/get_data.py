"""Implements dataloaders for the Ego4D dataset.

Here, the data is assumed to be in a folder titled "ego4d".
"""
import numpy as np
from torch.utils.data import DataLoader



def get_dataloader(data_dir, batch_size=40, num_workers=8, 
                   train_shuffle=True, 
                   flatten_imu=False,
                   unflatten_video=True, 
                   unsqueeze_channel=True, 
                   normalize_imu=True, 
                   normalize_video=True):
    """Get dataloaders for Ego4D.

    Args:
        data_dir (str): Directory of data.
        batch_size (int, optional): Batch size. Defaults to 40.
        num_workers (int, optional): Number of workers. Defaults to 8.
        train_shuffle (bool, optional): Whether to shuffle training data or not. Defaults to True.
        flatten_imu (bool, optional): Whether to flatten imu data or not. Defaults to False.
        unsqueeze_channel (bool, optional): Whether to unsqueeze any channels or not. Defaults to True.
        normalize_imu (bool, optional): Whether to normalize the imus before returning. Defaults to True.
        normalize_video (bool, optional): Whether to normalize the video before returning. Defaults to True.

    Returns:
        tuple: Tuple of (training dataloader, validation dataloader, test dataloader)
    """
    trains = [np.load(data_dir+"/imu/train_data.npy"), np.load(data_dir +
                                                                 "/video/train_data.npy"), np.load(data_dir+"/activity/train_data.npy")]
    tests = [np.load(data_dir+"/imu/test_data.npy"), np.load(data_dir +
                                                               "/video/test_data.npy"), np.load(data_dir+"/activity/test_data.npy")]
    print('trains:', trains[0].shape, trains[1].shape, trains[2].shape)
    # (1455, 100, 9) (1455, 48000) (1455,)
    print('tests:', tests[0].shape, tests[1].shape, tests[2].shape)
    # (162, 100, 9) (162, 48000) (162,)
    if flatten_imu:
        trains[0] = trains[0].reshape(1455, 100*9)
        tests[0] = tests[0].reshape(162, 100*9)
    if unflatten_video:
        trains[1] = trains[1].reshape(1455, 100, 480)
        tests[1] = tests[1].reshape(162, 100, 480)
    if normalize_imu:
        trains[0] /= 255.0
        tests[0] /= 255.0
    if normalize_video:
        trains[1] = trains[1]/255.0
        tests[1] = tests[1]/255.0
    if not normalize_imu:
        trains[0] = trains[0].reshape(60000, 100, 9)
        tests[0] = tests[0].reshape(10000, 100, 9)
    if unsqueeze_channel:
        trains[0] = np.expand_dims(trains[0], 1)
        tests[0] = np.expand_dims(tests[0], 1)
        trains[1] = np.expand_dims(trains[1], 1)
        tests[1] = np.expand_dims(tests[1], 1)
    trains[2] = trains[2].astype(int)
    tests[2] = tests[2].astype(int)
    trainlist = [[trains[j][i] for j in range(3)] for i in range(1455)]
    testlist = [[tests[j][i] for j in range(3)] for i in range(162)]
    valids = DataLoader(trainlist[1335:1455], shuffle=False,
                        num_workers=num_workers, batch_size=batch_size)
    tests = DataLoader(testlist, shuffle=False,
                       num_workers=num_workers, batch_size=batch_size)
    trains = DataLoader(trainlist[0:1335], shuffle=train_shuffle,
                        num_workers=num_workers, batch_size=batch_size, drop_last=True)
    return trains, valids, tests

