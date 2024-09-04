import os.path as osp
import subprocess

from scripts.script_config import pcqm_build_path


# https://github.com/MEPP-team/PCQM  Also see BUILD.md
def pcqm(infile1: str, infile2: str, omp_num_threads: int = None) -> float:
    subp_stdout = subprocess.run(
        f'{f"export OMP_NUM_THREADS={omp_num_threads};" if omp_num_threads is not None else ""}'
        f'cd {pcqm_build_path}; ./PCQM {osp.abspath(infile1)} {osp.abspath(infile2)} -fq',
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, check=True, text=True
    ).stdout
    for line in subp_stdout.splitlines():
        if line.startswith('PCQM value is : '):
            return float(line.rstrip().rsplit(' ', 1)[1])
