import os.path as osp
from typing import Tuple, Dict, Callable, Union, List


basic_types = Union[float, int, str]
extracted_log_type = Dict[str, basic_types]
log_mappings_type = Dict[str, Tuple[str, Callable[[str], basic_types]]]
all_file_metric_dict_type = Dict[str, Dict[str, List[basic_types]]]
one_file_metric_dict_type = Dict[str, List[basic_types]]


class LogExtractor:
    def __init__(self):
        pass

    def extract_log(self, log: str, mappings: log_mappings_type) -> extracted_log_type:
        lines = log.splitlines()
        extracted = {}
        for key, (new_key, map_fn) in mappings.items():
            for idx, line in enumerate(lines):
                if line.startswith(key):
                    extracted[new_key] = map_fn(line)
                    lines = lines[idx + 1:]
                    break
        return extracted


def read_file_list_with_rel_path(file_list):
    file_paths = []
    root_path = osp.split(file_list)[0]
    with open(file_list) as f:
        for line in f:
            file_path = osp.join(root_path, line.strip())
            if file_path not in file_paths:
                file_paths.append(file_path)
    return file_paths


def concat_values_for_dict(
        a: one_file_metric_dict_type,
        b: Union[Dict[str, Union[int, float, str]], one_file_metric_dict_type],
        use_default_value_for_b: bool = True,
        default_value=None
):
    if use_default_value_for_b:
        for key in a:
            if key not in b:
                b[key] = default_value
    for key, value in b.items():
        if key in a:
            if not isinstance(a[key], List):
                a[key] = [a[key]]
            if isinstance(value, List):
                a[key].extend(value)
            else:
                a[key].append(value)
        else:
            if isinstance(value, List):
                a[key] = value
            else:
                a[key] = [value]

    return a


def concat_values_for_dict_2(
        a: all_file_metric_dict_type,
        b: all_file_metric_dict_type
):
    for key, value in b.items():
        concat_values_for_dict(a[key], b[key])
