# DiffSF: Diffusion Models for Scene Flow Estimation
### [Project Page (TODO)]() | [Paper](https://arxiv.org/abs/2403.05327)
<br/>

> DiffSF: Diffusion Models for Scene Flow Estimation
> [Yushan Zhang](https://scholar.google.com/citations?user=mvY4rdIAAAAJ&hl=en), [Bastian Wandt](https://scholar.google.com/citations?user=z4aXEBYAAAAJ), [Maria Magnusson](), [Michael Felsberg](https://scholar.google.com/citations?&user=lkWfR08AAAAJ)  
> Arxiv 2024

## Get started

# Installation:
Create a conda environment:
```bash
conda create -n diffsf python=3.9
conda activate diffsf
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```
Install other dependencies:
```bash
pip install tqdm tensorboard opencv-python
```

# Dataset Preparation:

1. FlyingThings3D(HPLFlowNet without occlusion / CamLiFlow with occlusion):

Download [FlyingThings3D_subset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html).
flyingthings3d_disparity.tar.bz2, flyingthings3d_disparity_change.tar.bz2, FlyingThings3D_subset_disparity_occlusions.tar.bz2, FlyingThings3D_subset_flow.tar.bz2, FlyingThings3D_subset_flow_occlusions.tar.bz2 and FlyingThings3D_subset_image_clean.tar.bz2 are needed. Then extract the files in /path/to/flyingthings3d such that the directory looks like
```bash
/path/to/flyingthings3d
├── train/
│   ├── disparity/
│   ├── disparity_change/
│   ├── disparity_occlusions/
│   ├── flow/
│   ├── flow_occlusions/
│   ├── image_clean/
├── val/
│   ├── disparity/
│   ├── disparity_change/
│   ├── disparity_occlusions/
│   ├── flow/
│   ├── flow_occlusions/
│   ├── image_clean/
```
Preprocess dataset using the following command:
```bash
cd utils
python preprocess_flyingthings3d_subset.py --input_dir /mnt/data/flyingthings3d_subset --output_dir flyingthings3d_subset
python preprocess_flyingthings3d_subset.py --input_dir /mnt/data/flyingthings3d_subset --output_dir flyingthings3d_subset_non-occluded --remove_occluded_points
```

2. FlyingThings3D(FlowNet3D with occlusion):

The processed data is also provided [here](https://drive.google.com/file/d/1CMaxdt-Tg1Wct8v8eGNwuT7qRSIyJPY-/view?usp=sharing) for download (total size ~11GB)

3. KITTI(HPLFlowNet without occlusion):

First, download the following parts:
Main data: [data_scene_flow.zip](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip)
Calibration files: [data_scene_flow_calib.zip](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow_calib.zip)
Unzip them and organize the directory as follows:
```bash
datasets/KITTI_stereo2015
├── testing
│   ├── calib_cam_to_cam
│   ├── calib_imu_to_velo
│   ├── calib_velo_to_cam
│   ├── image_2
│   ├── image_3
└── training
    ├── calib_cam_to_cam
    ├── calib_imu_to_velo
    ├── calib_velo_to_cam
    ├── disp_noc_0
    ├── disp_noc_1
    ├── disp_occ_0
    ├── disp_occ_1
    ├── flow_noc
    ├── flow_occ
    ├── image_2
    ├── image_3
    ├── obj_map
```
Preprocess dataset using the following command:
```bash
cd utils
python process_kitti.py datasets/KITTI_stereo2015/ SAVE_PATH/KITTI_processed_occ_final
```

4. KITTI(FlowNet3D with occlusion):

The processed data is also provided [here](https://drive.google.com/open?id=1XBsF35wKY0rmaL7x7grD_evvKCAccbKi) for download

5. Waymo-Open(refer to [FH-Net](https://github.com/pigtigger/FH-Net)):
Download the Waymo raw data from [link_to_waymo_open_dataset](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_4_0;tab=objects?pli=1&prefix=&forceOnObjectsSortingFiltering=false),run the following command to extract point clouds, 3D annotations, poses and other information form raw data.
```bash
cd diffsf
python waymo_tools/waymo_extract.py

```
After extracting data, the folder structure is the same as below:
```bash
datasets
├── waymo-open
│   ├── scene_flow
│       ├── ImageSets
│       ├── train
│       ├── valid
├── train_extract
│   ├── 000
│   ├── 001
│   ├── ...
├── valid_extract
│   ├── 000
│   ├── 001
│   ├── ...
```
Then create scene flow data by:
```bash
python waymo_tools/create_data.py --dataset_type waymo
```

# The datasets directory should be orginized as:
```bash
datasets
├── datasets_KITTI_flownet3d
│   ├── kitti_rm_ground
├── datasets_KITTI_hplflownet
│   ├── KITTI_processed_occ_final
├── FlyingThings3D_flownet3d
├── flyingthings3d_subset
│   ├── train
│   ├── val
├── flyingthings3d_subset_non-occluded
│   ├── train
│   ├── val
├── KITTI_stereo2015
│   ├── testing
│   ├── training
├── waymo-open
│   ├── scene_flow
│   ├── train_extract
│   ├── valid_extract
```

# Traning and Testing:
1-FlyThings3D_occ
Training (FlyThings3D_occ): 
```bash
python diffsf_main.py --train_dataset f3d_occ --lr 4e-4 --train_batch_size 24
```
Testing (FlyThings3D_occ): 
```bash
python diffsf_main.py --val_dataset f3d_occ --eval
Testing (KITTI_occ): 
```
```bash
python diffsf_main.py --val_dataset kitti_occ --eval
```
2-FlyThings3D_nonocc
Training (FlyThings3D_nonocc): 
```bash
python diffsf_main.py --train_dataset f3d_nonocc --lr 4e-4 --train_batch_size 24
```
Testing (FlyThings3D_nonocc): 
```bash
python diffsf_main.py --val_dataset f3d_nonocc --eval
Testing (KITTI_nonocc): 
```
```bash
python diffsf_main.py --val_dataset kitti_nonocc --eval
```
3-Waymo-Open
Training (Waymo-Open): 
```bash
python diffsf_main.py --train_dataset waymo --lr 1e-4 --train_batch_size 24
```
Testing (Waymo-Open): 
```bash
python diffsf_main.py --val_dataset waymo --eval
```

# Pretrained Checkpoints
1-[FlyThings3D_occ](https://drive.google.com/file/d/1t0jx65qHTym4zdgrXIBcXxYybfyqFuvs/view?usp=sharing)
2-[FlyThings3D_nonocc](https://drive.google.com/file/d/14vVhReQ_canUdhqHqGeOoK0GqUqaFT0S/view?usp=sharing)
3-[Waymo-Open](https://drive.google.com/file/d/1bfTUjdWXmFQxlrX5SWAgRnUWP58NcWzp/view?usp=sharing)

## BibTeX
If you find our models useful, please consider citing our paper!
```
@article{zhang2024diffsf,
  title={DiffSF: Diffusion Models for Scene Flow Estimation},
  author={Zhang, Yushan and Wandt, Bastian and Magnusson, Maria and Felsberg, Michael},
  journal={arXiv preprint arXiv:2403.05327},
  year={2024}
}
```