import os
import re
import datetime

import numpy as np


def make_new_dirs(dir_path, logger) -> None:
    if os.path.exists(dir_path):
        logger.warning('folder "{}" ready exists.'.format(dir_path))
        for obj_name in os.listdir(dir_path):
            obj_path = os.path.join(dir_path, obj_name)
            if os.path.isfile(obj_path):
                new_name = obj_name[:obj_name.rfind('.')] + '_bak' + obj_name[obj_name.rfind('.'):]
            else:
                new_name = obj_name + '_bak'
            new_path = os.path.join(dir_path, new_name)
            os.rename(obj_path, new_path)
            logger.warning(f'rename {obj_path} as {new_path}')
    else:
        os.makedirs(dir_path)
        logger.info('make dirs "{}"'.format(dir_path))


def autoindex_obj(obj_path: str) -> str:
    dir_path, obj_name = os.path.split(obj_path)
    notations = {
        '<maxindex>': lambda x: max(x + [0]),
        '<minindex>': lambda x: min(x + [0]),
        '<autoindex>': lambda x: max(x + [-1]) + 1,
    }
    if not os.path.exists(dir_path):
        for notation in notations:
            obj_path = obj_path.replace(notation, '0')
        return obj_path
    for notation in notations:
        if obj_name.find(notation) != -1:
            pattern = obj_name.replace(notation, '([0-9]+)')
            objects_exist = os.listdir(dir_path)
            indexes_exist = []

            for name in objects_exist:
                match_res = re.match(pattern, name)
                if match_res:
                    indexes_exist.append(int(match_res.group(1)))

            obj_path = obj_path.replace(notation, str(notations[notation](indexes_exist)))
            break
    return obj_path


def totaltime_by_seconds(seconds, no_microseconds=True):
    total_time = datetime.timedelta(seconds=seconds)
    if no_microseconds:
        total_time = total_time - datetime.timedelta(microseconds=total_time.microseconds)
    return total_time


def eta_by_seconds(seconds, no_microseconds=True):
    time_now = datetime.datetime.now()
    total_time = datetime.timedelta(seconds=seconds)
    if no_microseconds:
        total_time = total_time - datetime.timedelta(microseconds=total_time.microseconds)
        time_now = time_now - datetime.timedelta(microseconds=time_now.microsecond)
    return total_time, time_now + total_time


def entropy(*num_list: int) -> float:
    num_list = np.array(num_list)
    num_list_non_zero = num_list[num_list.nonzero()]
    freq_list = num_list_non_zero / num_list_non_zero.sum()
    return ((-freq_list * np.log2(freq_list)).sum() * num_list.sum()).item()


if __name__ == '__main__':
    pass
