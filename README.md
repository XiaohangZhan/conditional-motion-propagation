# Implementation of "Self-Supervised Learning via Conditional Motion Propagation" (CMP)

### Paper

Xiaohang Zhan, Xingang Pan, Ziwei Liu, Dahua Lin, Chen Change Loy, "[Self-Supervised Learning via Conditional Motion Propagation](https://arxiv.org/abs/1903.11412)", in CVPR 2019 [[Project Page](http://mmlab.ie.cuhk.edu.hk/projects/CMP/)]

For further information, please contact [Xiaohang Zhan](https://xiaohangzhan.github.io/).

### Demos

* Conditional motion propagation

* Guided video generation (draw arrows to let a static image animated)

![](demos/demo_video_generation.gif)

* Semi-automatic annotation (first row: interface, auto zoom-in, mask; second row: optical flows)

![](demos/demo_annotation.gif)

### Data collection

[YFCC frames](https://dl.fbaipublicfiles.com/unsupervised-video/UnsupVideo_Frames_v1.tar.gz) (45G).
[YFCC optical flows (LiteFlowNet)](https://drive.google.com/open?id=1S_TU1UjKms-U_Q4bOhXfUfIJX5hgwOtq) (29G).
[YFCC lists](https://drive.google.com/open?id=1ObzO7xWXolPKrIC39XCvjttZYEoVn6k2) (251M).

### Model collection

* Comming soon

### Requirements
 
* python>=3.6
* pytorch>=0.4.0
* others

```shell
pip install -r requirements.txt
```

### Usage

0. Clone the repo.

    ```shell
    git clone git@github.com:XiaohangZhan/conditional-motion-propagation.git
    cd conditional-motion-propagation
    ```

1. Prepare data (YFCC as an example)

    ```shell
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

    ```shell
    sh experiments/rep_learning/alexnet_yfcc/train.sh # 8 GPUs distributed training
    sh tools/weight_process.sh --config experiments/rep_learning/alexnet_yfcc/config.yaml --iter 70000 # extract weights of the image encoder
    ```

3. Train CMP for Video generation and Semi-automatic Annotation.

    ```shell
    # comming soon
    ```

4. Run demos

    ```shell
    # comming soon
    ```

### Results

### Bibtex

```
@inproceedings{zhan2019conditional,
 author = {Zhan, Xiaohang and Pan, Xingang and Liu, Ziwei and Lin, Dahua and Loy, Chen Change},
 title = {Self-Supervised Learning via Conditional Motion Propagation},
 booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR)},
 month = {June},
 year = {2019}
}
```
