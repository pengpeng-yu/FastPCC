import os
import subprocess


def gpcc_octree_lossless_geom_encode(in_path, out_path, command='tmc3'):
    args = ' --mode=0' \
        ' --trisoupNodeSizeLog2=0' \
        ' --mergeDuplicatedPoints=0' \
        ' --neighbourAvailBoundaryLog2=8' \
        ' --intra_pred_max_node_size_log2=6' \
        ' --positionQuantizationScale=1' \
        ' --inferredDirectCodingMode=1' \
        ' --maxNumQtBtBeforeOt=4' \
        ' --minQtbtSizeLog2=0' \
        ' --planarEnabled=1' \
        ' --planarModeIdcmUse=0' \
        f' --uncompressedDataPath={in_path}' \
        f' --compressedStreamPath={out_path}'
    cmd_args = command + args
    subp_stdout = subprocess.run(
        cmd_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        shell=True, check=True, text=True
    )
    if not os.path.isfile(out_path):
        raise RuntimeError(subp_stdout)


def gpcc_decode(in_path, out_path, command='tmc3'):
    args = ' --mode=1' \
        f' --compressedStreamPath={in_path}' \
        f' --reconstructedDataPath={out_path}' \
        ' --outputBinaryPly=0'
    cmd_args = command + args
    subp_stdout = subprocess.run(
        cmd_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        shell=True, check=True, text=True
    )
    if not os.path.isfile(out_path):
        raise RuntimeError(subp_stdout)
