from pyntcloud import PyntCloud


def main():
    path = r'I:\data\ModelNet\modelnet40_normal_resampled\airplane\airplane_0001.txt'
    pc:PyntCloud = PyntCloud.from_file(path, header=None, names=['x', 'y', 'z', '?1', '?2', '?3'])
    pc.get_sample()
    pc.plot(backend='pyvista')
    print('Done')


if __name__ == '__main__':
    main()