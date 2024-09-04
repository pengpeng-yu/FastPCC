import os.path as osp
import subprocess

from scripts.script_config import graph_sim_project_path, matlab_path


# https://github.com/NJUVISION/GraphSIM  Also see BUILD.md
def graph_sim(infile1: str, infile2: str, omp_num_threads: int = None) -> float:
    subp_stdout = subprocess.run(
        f'{f"export OMP_NUM_THREADS={omp_num_threads};" if omp_num_threads is not None else ""}'
        f'cd {graph_sim_project_path}; '
        f'{osp.abspath(matlab_path)} -nodisplay -nosplash -batch '
        f'"cd gspbox-master; gsp_start; cd ..;'
        f'global name_r;global name_d;'
        f'name_r = (\'{osp.abspath(infile1)}\');'
        f'name_d = (\'{osp.abspath(infile2)}\');'
        f'demo_fast_make; GraphSIM; exit"',
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, check=True, text=True
    ).stdout
    for line in subp_stdout.splitlines():
        if line.startswith('GraphSIM: '):
            return float(line.rstrip().rsplit(' ', 1)[1])
