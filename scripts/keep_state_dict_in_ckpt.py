from glob import glob
import os.path as osp
import torch


def keep_state_dict_in_ckpt(dir_path):
    files = glob(osp.join(dir_path, '**', '*.pt'), recursive=True)
    for file in files:
        ckpt = torch.load(file, map_location='cpu')
        if len(ckpt.keys()) != 1:
            ckpt = {'state_dict': ckpt['state_dict']}
            torch.save(ckpt, file)
        else:
            assert list(ckpt.keys())[0] == 'state_dict'


if __name__ == '__main__':
    keep_state_dict_in_ckpt('weights')
