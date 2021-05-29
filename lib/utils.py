import os
import re
import datetime


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
        '<maxindex>': lambda x: max(x),
        '<minindex>': lambda x: min(x),
        '<autoindex>': lambda x: max(x) + 1,
    }
    if not os.path.exists(dir_path):
        for notaion in notations:
            obj_path = obj_path.replace(notaion, '0')
        return obj_path
    for notaion in notations:
        if obj_name.find(notaion) != -1:
            pattern = obj_name.replace(notaion, '([0-9]+)')
            objects_exist = os.listdir(dir_path)
            indexes_exist = [-1]

            for name in objects_exist:
                match_res = re.match(pattern, name)
                if match_res:
                    indexes_exist.append(int(match_res.group(1)))

            obj_path = obj_path.replace(notaion, str(notations[notaion](indexes_exist)))
            break
    return obj_path


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
