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
    command += args
    subprocess.run(command, shell=True, check=True)


def gpcc_decode(in_path, out_path, command='tmc3'):
    args = ' --mode=1' \
        f' --compressedStreamPath={in_path}' \
        f' --reconstructedDataPath={out_path}' \
        ' --outputBinaryPly=0'
    command += args
    subprocess.run(command, shell=True, check=True)
