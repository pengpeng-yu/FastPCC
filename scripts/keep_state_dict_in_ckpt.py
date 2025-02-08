from glob import glob
import os.path as osp
import torch


def keep_state_dict_in_ckpt(dir_path):
    files = glob(osp.join(dir_path, '**', '*.pt'), recursive=True)
    for file in files:
        ckpt = torch.load(file, map_location='cpu')
        if len(ckpt.keys()) != 1:
            key = 'ema_state_dict'
            if key not in ckpt: key = 'state_dict'
            torch.save({key: ckpt[key]}, file)
        else:
            key = list(ckpt.keys())[0]
            assert key == 'state_dict' or key == 'ema_state_dict', file


if __name__ == '__main__':
    keep_state_dict_in_ckpt('weights')
