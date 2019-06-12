# Implementation of "Self-Supervised Learning via Conditional Motion Propagation" (CMP)

## Paper

Xiaohang Zhan, Xingang Pan, Ziwei Liu, Dahua Lin, Chen Change Loy, "[Self-Supervised Learning via Conditional Motion Propagation](https://arxiv.org/abs/1903.11412)", in CVPR 2019 [[Project Page](http://mmlab.ie.cuhk.edu.hk/projects/CMP/)]

For further information, please contact [Xiaohang Zhan](https://xiaohangzhan.github.io/).

## Demos (Watching full demos in [YouTube](https://www.youtube.com/watch?v=6R_oJCq5qMw))

* Conditional motion propagation

* Guided video generation (draw arrows to let a static image animated)

![](demos/demo_video_generation.gif)

* Semi-automatic annotation (first row: interface, auto zoom-in, mask; second row: optical flows)

![](demos/demo_annotation.gif)

## Data collection

[YFCC frames](https://dl.fbaipublicfiles.com/unsupervised-video/UnsupVideo_Frames_v1.tar.gz) (45G).
[YFCC optical flows (LiteFlowNet)](https://drive.google.com/open?id=1S_TU1UjKms-U_Q4bOhXfUfIJX5hgwOtq) (29G).
[YFCC lists](https://drive.google.com/open?id=1ObzO7xWXolPKrIC39XCvjttZYEoVn6k2) (251M).

## Model collection

* Pre-trained models for semantic segmentation, instance segmentation and human parsing by CMP can be downloaded [here](https://drive.google.com/open?id=1Kx-OIcr2U44p9mlpV-SbhANQdtbn2rJR)

* Models for demos (conditinal motion propagation, guided video generation and semi-automatic annotation) can be downloaded [here](https://drive.google.com/open?id=1JMuoexvRCUQ0cmtfyse-8OScLHA6tjuI)

## Requirements
 
* python>=3.6
* pytorch>=0.4.0
* others

    ```sh
    pip install -r requirements.txt
    ```

## Usage

0. Clone the repo.

    ```sh
    git clone git@github.com:XiaohangZhan/conditional-motion-propagation.git
    cd conditional-motion-propagation
    ```

### Representation learning

1. Prepare data (YFCC as an example)

    ```sh
    mkdir data
    mkdir data/yfcc
    cd data/yfcc
    # download YFCC frames, optical flows and lists to data/yfcc
    tar -xf UnsupVideo_Frames_v1.tar.gz
    tar -xf flow_origin.tar.gz
    tar -xf lists.tar.gz
    ```
    Then folder `data` looks like:
    ```
    data
      ├── yfcc
        ├── UnsupVideo_Frames
        ├── flow_origin
        ├── lists
          ├── train.txt
          ├── val.txt
    ```

2. Train CMP for Representation Learning.

    * If your server supports multi-nodes training.

    ```sh
    sh experiments/rep_learning/alexnet_yfcc_16gpu_70k/train.sh # 16 GPUs distributed training
    python tools/weight_process.py --config experiments/rep_learning/alexnet_yfcc_16gpu_70k/config.yaml --iter 70000 # extract weights of the image encoder to experiments/rep_learning/alexnet_yfcc_16gpu_70k/checkpoints/convert_iter_70000.pth.tar
    ```

    * If your server does not support multi-nodes training.
    ```sh
    sh experiments/rep_learning/alexnet_yfcc_8gpu_140k/train.sh # 8 GPUs distributed training
    python tools/weight_process.py --config experiments/rep_learning/alexnet_yfcc_8gpu_140k/config.yaml --iter 140000 # extract weights of the image encoder
    ```

### Run demos

1. Download the model and move it to `experiments/semiauto_annot/resnet50_vip+mpii_liteflow/checkpoints/`

2. Launch jupyter notebook and run `demos/cmp.ipynb` for conditional motion propagation, or `demos/demo_annot.ipynb` for semi-automatic annotation.

3. Train the model by yourself (optional)

    ```sh
    # data not ready
    sh experiments/semiauto_annot/resnet50_vip+mpii_liteflow/train.sh # 8 GPUs distributed training
    ```

### Results

### Bibtex

```
@inproceedings{zhan2019self,
 author = {Zhan, Xiaohang and Pan, Xingang and Liu, Ziwei and Lin, Dahua and Loy, Chen Change},
 title = {Self-Supervised Learning via Conditional Motion Propagation},
 booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR)},
 month = {June},
 year = {2019}
}
```
