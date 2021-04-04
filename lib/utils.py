import os
import re
import datetime


def make_new_dirs(dir_path, logger) -> None:
    if os.path.exists(dir_path):
        logger.critical('dir "{}" ready exists'.format(dir_path))
        raise FileExistsError
    else:
        os.makedirs(dir_path)
        logger.info('make dirs "{}"'.format(dir_path))


def auto_index_dir(path, dir_name:str) -> str:
    if dir_name.find('<autoindex>') != -1:
        index = 0
        dir_pattern = dir_name.replace('<autoindex>', '([0-9]+)')
        dirs_exist = os.listdir(path)

        for name in dirs_exist:
            if re.match(dir_pattern, name):
                index_exists =  int(re.match(dir_pattern, name).group(1))
                if index_exists >= index: index = index_exists + 1
        dir_name = dir_name.replace('<autoindex>', str(index))
    return os.path.join(path, dir_name)


def totaltime_by_seconds(seconds, no_microseconds=True):
    totall_time = datetime.timedelta(seconds=seconds)
    if no_microseconds:
        totall_time = totall_time - datetime.timedelta(microseconds=totall_time.microseconds)
    return totall_time


def eta_by_seconds(seconds, no_microseconds=True):
    time_now = datetime.datetime.now()
    totall_time = datetime.timedelta(seconds=seconds)
    if no_microseconds:
        totall_time = totall_time - datetime.timedelta(microseconds=totall_time.microseconds)
        time_now = time_now - datetime.timedelta(microseconds=time_now.microsecond)
    return totall_time, time_now + totall_time


if __name__ == '__main__':
    pass
