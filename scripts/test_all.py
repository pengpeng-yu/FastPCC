from glob import glob
import subprocess


def test_all(rundir_name, config_paths, glob_weights_paths):
    for config_path, glob_weights_path in zip(config_paths, glob_weights_paths):
        weights_path = glob(glob_weights_path)
        for weight_path in weights_path:
            subprocess.run(
                '/bin/bash -c "source /home/xiaoyumike/miniconda3/bin/activate py37torch110;'
                f'python test.py {config_path}'
                f' test.weights_from_ckpt={weight_path}'
                f' test.rundir_name={rundir_name}/{weight_path.split("/", 1)[1].rsplit(".", 1)[0]}"',
                shell=True, check=True
            )


if __name__ == '__main__':
    rundir_name = 'test_all'
    glob_weights_paths = [
        'weights/lossless_based/*.pt',
        'weights/hyperprior_factorized/*.pt',
        'weights/hyperprior_scale_normal/*.pt',
        'weights/baseline/*.pt',
        'weights/baseline_4x/*.pt'
    ]
    config_paths = [
        'configs/train/convolutional/lossless_based.yaml',
        'configs/train/convolutional/hyperprior_factorized',
        'configs/train/convolutional/hyperprior_scale_normal',
        'configs/train/convolutional/baseline',
        'configs/train/convolutional/baseline_4x.yaml'
    ]
    test_all(rundir_name, config_paths, glob_weights_paths)
