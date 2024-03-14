from datasetloader.kitti import KITTI_flownet3d, KITTI_hplflownet
from datasetloader.waymo import Waymo
from datasetloader.flyingthings3d import FlyingThings3D_subset, FlyingThings3D_flownet3d

def build_train_dataset(dataset):

    if dataset == 'f3d_nonocc':
        train_dataset = FlyingThings3D_subset(split='train', occ=False, npoints=4096)
    elif dataset == 'f3d_occ':
        train_dataset = FlyingThings3D_flownet3d(train=True, npoints=4096)
    elif dataset == 'waymo':
        train_dataset = Waymo(split='train', npoints=4096)
    else:
        raise ValueError(f'stage {dataset} is not supported')

    return train_dataset

def build_test_dataset(dataset):

    if dataset == 'f3d_nonocc':
        test_dataset = FlyingThings3D_subset(split='val', occ=False, npoints=8192)
    elif dataset == 'f3d_occ':
        test_dataset = FlyingThings3D_flownet3d(train=False, npoints=8192)
    elif dataset == 'kitti_nonocc':
        test_dataset = KITTI_hplflownet(train=False)
    elif dataset == 'kitti_occ':
        test_dataset = KITTI_flownet3d(split='training150')
    elif dataset == 'waymo':
        test_dataset = Waymo(split='valid', npoints=8192)
    else:
        raise ValueError(f'stage {dataset} is not supported')

    return test_dataset