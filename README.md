<h2 align="center"> <a href="https://arxiv.org/abs/2503.10624">ETCH: Generalizing Body Fitting to Clothed Humans via Equivariant Tightness</a>
</h2>

<h3 align="center">
International Conference on Computer Vision 2025 (ICCV 2025)
</h3>

<h4 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2503.10624-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2503.10624)
[![Home Page](https://img.shields.io/badge/Project-Website-green.svg)](https://boqian-li.github.io/ETCH/) 
[![X](https://img.shields.io/badge/@Boqian%20Li-black?logo=X)](https://x.com/Boqian_Li_/status/1908467186122817642)
[![youtube](https://img.shields.io/badge/Video-E33122?logo=Youtube)](https://youtu.be/8_3DdW0cZqM)

[Boqian Li](https://boqian-li.github.io/), 
[Haiwen Feng](https://havenfeng.github.io/), 
[Zeyu Cai](https://github.com/zcai0612), 
[Michael J. Black](https://ps.is.mpg.de/person/black), 
[Yuliang Xiu](https://xiuyuliang.cn/) 
</h4>

This repository is the official implementation of ETCH, a novel pipeline that estimates cloth-to-body surface mapping through locally approximate SE(3) equivariance, encoding tightness as displacement vectors from the cloth surface to the underlying body.

<h2 align="center">
This is dev branch.
</h2>



## Environment Setup

```bash
conda env create -f environment.yml
conda activate etch
cd external
git clone https://github.com/facebookresearch/theseus.git && cd theseus
pip install -e .
cd ../..
```

## Data Preparation
0. please note that we placed data samples in the `datafolder` folder for convenience. 
1. Generate Anchor Points with Tightness Vectors (for training)
    ```bash
    python scripts/generate_infopoints.py
    ```

2. Get splitted ids (pkl file)
    ```bash
    python scripts/get_splitted_ids_{datasetname}.py
    ```
3. For body_models, please download with [this link](https://drive.google.com/file/d/1JNFk4OGfDkgE9WdJb1D1zGaECix8XpKV/view?usp=sharing).

4. please note that before the above processes: 

    for `4D-Dress` dataset, we apply zero-translation `mesh.apply_translation(-translation)` to the scan and the body model; 
    
    for `CAPE` dataset, we used the processed meshes extracted from [PTF](https://github.com/taconite/PTF), in which we noticed that the SMPL body meshes are marginally different from the original SMPL body meshes but more precise.

## Dataset Organization
The dataset folder tree is like:  
```bash
datafolder/
├── datasetfolder/
│   ├── model/ # scans
│   │   ├── id_0
│   │   │   └── id_0.obj
│   ├── smpl(h)/ # body models
│   │   ├── id_0
│   │   │   ├── info_id_0.npz
│   │   │   └── mesh_smpl_id_0.obj # SMPL body mesh
├── useful_data_datasetname/
├── gt_datasetname_data/
│   ├── npz/
│   │   └── id_0.npz
│   └── ply 
│       └── id_0.ply
```
please refer to the `datafolder` folder for more details. 


## Training

```bash
CUDA_VISIBLE_DEVICES=0 python src/train.py --batch_size 2 --i datasetname_settingname
```

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python src/eval.py --batch_size 3 --model_path model_path --i datasetname_settingname
```

## Pretrained Model
Please download the pretrained model used in the paper from [here](https://drive.google.com/drive/folders/14zGMkmC580VLNgeUBFtM6FP8QX415VAa?usp=sharing). 