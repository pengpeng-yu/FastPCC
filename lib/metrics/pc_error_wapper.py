import subprocess
from typing import Dict


_DIVIDERS = ['1. Use infile1 (A) as reference, loop over A, use normals on B. (A->B).',
             '2. Use infile2 (B) as reference, loop over B, use normals on A. (B->A).',
             '3. Final (symmetric).',
             'Job done!']


def mpeg_pc_error(
        infile1: str, infile2: str, resolution: int, normal_file: str = '',
        hausdorff: bool = False, color: bool = False, threads: int = 1, command='pc_error_d'
) -> Dict[str, float]:
    cmd_args = f'{command}' \
               f' -a {infile1}' \
               f' -b {infile2}' \
               f' --resolution={resolution - 1}' \
               f' --hausdorff={int(hausdorff)}' \
               f' --color={int(color)}' \
               f' --nbThreads={threads}'
    if normal_file != '':
        cmd_args += ' -n ' + normal_file

    subp_stdout = subprocess.run(
        cmd_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        shell=True, check=True, text=True
    ).stdout

    metric_dict = {}
    flag_read = False
    for line in subp_stdout.splitlines():
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
