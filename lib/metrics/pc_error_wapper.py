import subprocess
from typing import Dict, Callable, Union, Tuple
import os.path as osp

from scripts.shared_config import pc_error_path


_DIVIDERS = ['1. Use infile1 (A) as reference, loop over A, use normals on B. (A->B).',
             '2. Use infile2 (B) as reference, loop over B, use normals on A. (B->A).',
             '3. Final (symmetric).',
             'Job done!']


def mpeg_pc_error(
        infile1: str, infile2: str, resolution: int, normal_file: str = '',
        hausdorff: bool = False, color: bool = False, threads: int = 1, command='',
        hooks: Tuple[Callable[[str], Tuple[str, Union[None, int, float, str]]]] = ()
) -> Dict[str, float]:
    if command == '': command = pc_error_path
    cmd_args = f'{command}' \
               f' -a {infile1}' \
               f' -b {infile2}' \
               f' --resolution={resolution - 1}' \
               f' --hausdorff={int(hausdorff)}' \
               f' --color={int(color)}' \
               f' --nbThreads={threads}'
    if normal_file != '' and osp.exists(normal_file):
        cmd_args += ' -n ' + normal_file

    subp_stdout = subprocess.run(
        cmd_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        shell=True, check=True, text=True
    ).stdout

    metric_dict = {}
    flag_read = False
    for line in subp_stdout.splitlines():
        for hook in hooks:
            extracted = hook(line)
            if extracted is not None:
                metric_dict[extracted[0]] = extracted[1]
        if line.startswith(_DIVIDERS[0]):
            flag_read = True
        elif line.startswith(_DIVIDERS[-1]):
            break
        elif flag_read and ':' in line:
            line = line.strip()
            key, value = line.split(':', 1)
            metric_dict[key.strip()] = float(value)

    if metric_dict == {}:
        raise RuntimeError(subp_stdout)
    return metric_dict
