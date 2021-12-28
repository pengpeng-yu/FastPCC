import os
import re
import datetime
import time

import numpy as np


def make_new_dirs(dir_path, logger):
    if os.path.exists(dir_path):
        logger.warning('folder "{}" ready exists.'.format(dir_path))
        target_path = autoindex_obj(str(dir_path) + '_bak<autoindex>')
        os.rename(dir_path, target_path)
        logger.warning(f'rename {dir_path} as {target_path}')
    os.makedirs(dir_path)
    logger.info('make dirs "{}"'.format(dir_path))


def autoindex_obj(obj_path: str) -> str:
    dir_path, obj_name = os.path.split(os.path.abspath(obj_path))
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
            pattern = re.compile(obj_name.replace(notation, '([0-9]+)') + '$')
            objects_exist = os.listdir(dir_path)
            indexes_exist = []

            for name in objects_exist:
                match_res = pattern.match(name)
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


class Timer:
    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_time = time.time() - self.start_time
        return False


if __name__ == '__main__':
    pass
