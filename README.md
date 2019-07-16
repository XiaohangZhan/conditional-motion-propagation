# Implementation of "Self-Supervised Learning via Conditional Motion Propagation" (CMP)

## Paper

Xiaohang Zhan, Xingang Pan, Ziwei Liu, Dahua Lin, Chen Change Loy, "[Self-Supervised Learning via Conditional Motion Propagation](https://arxiv.org/abs/1903.11412)", in CVPR 2019 [[Project Page](http://mmlab.ie.cuhk.edu.hk/projects/CMP/)]

For further information, please contact [Xiaohang Zhan](https://xiaohangzhan.github.io/).

## Demos (Watching full demos in [YouTube](https://www.youtube.com/watch?v=6R_oJCq5qMw))

* Conditional motion propagation (motion prediction by guidance)

![](demos/demo_cmp.gif)

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
    sh experiments/rep_learning/alexnet_yfcc_voc_16gpu_70k/train.sh # 16 GPUs distributed training
    python tools/weight_process.py --config experiments/rep_learning/alexnet_yfcc_voc_16gpu_70k/config.yaml --iter 70000 # extract weights of the image encoder to experiments/rep_learning/alexnet_yfcc_voc_16gpu_70k/checkpoints/convert_iter_70000.pth.tar
    ```

    * If your server does not support multi-nodes training.
    ```sh
    sh experiments/rep_learning/alexnet_yfcc_voc_8gpu_140k/train.sh # 8 GPUs distributed training
    python tools/weight_process.py --config experiments/rep_learning/alexnet_yfcc_voc_8gpu_140k/config.yaml --iter 140000 # extract weights of the image encoder
    ```

### Run demos

1. Download the [model](https://drive.google.com/open?id=1JMuoexvRCUQ0cmtfyse-8OScLHA6tjuI) and move it to `experiments/semiauto_annot/resnet50_vip+mpii_liteflow/checkpoints/`.

2. Launch jupyter notebook and run `demos/cmp.ipynb` for conditional motion propagation, or `demos/demo_annot.ipynb` for semi-automatic annotation.

3. Train the model by yourself (optional)

    ```sh
    # data not ready
    sh experiments/semiauto_annot/resnet50_vip+mpii_liteflow/train.sh # 8 GPUs distributed training
    ```

### Results

<h4>1. Pascal VOC 2012 Semantic Segmentation (AlexNet)</h4>
    <table class="table table-condensed">
        <th>Method (AlexNet)</th><th>Supervision (data amount)</th><th>% mIoU</th>
        <tbody>
        <tr><td>Krizhevsky et al. [1]</td><td>ImageNet labels (1.3M)</td><td>48.0</td></tr>
        <tr><td>Random</td><td>- (0)</td><td>19.8</td></tr>
        <tr><td>Pathak et al. [2]</td><td>In-painting (1.2M)</td><td>29.7</td></tr>
        <tr><td>Zhang et al. [3]</td><td>Colorization (1.3M)</td><td>35.6</td></tr>
        <tr><td>Zhang et al. [4]</td><td>Split-Brain (1.3M)</td><td>36.0</td></tr>
        <tr><td>Noroozi et al. [5]</td><td>Counting (1.3M)</td><td>36.6</td></tr>
        <tr><td>Noroozi et al. [6]</td><td>Jigsaw (1.3M)</td><td>37.6</td></tr>
        <tr><td>Noroozi et al. [7]</td><td>Jigsaw++ (1.3M)</td><td>38.1</td></tr>
        <tr><td>Jenni et al. [8]</td><td>Spot-Artifacts (1.3M)</td><td>38.1</td></tr>
        <tr><td>Larsson et al. [9]</td><td>Colorization (3.7M)</td><td>38.4</td></tr>
        <tr><td>Gidaris et al. [10]</td><td>Rotation (1.3M)</td><td>39.1</td></tr>
        <tr><td>Pathak et al. [11]*</td><td>Motion Segmentation (1.6M)</td><td>39.7</td></tr>
        <tr><td>Walker et al. [12]*</td><td>Flow Prediction (3.22M)</td><td>40.4</td></tr>
        <tr><td>Mundhenk et al. [13]</td><td>Context (1.3M)</td><td>40.6</td></tr>
        <tr><td>Mahendran et al. [14]</td><td>Flow Similarity (1.6M)</td><td>41.4</td></tr>
        <tr><td>Ours</td><td>CMP (1.26M)</td><td>42.9</td></tr>
        <tr><td>Ours</td><td>CMP (3.22M)</td><td>44.5</td></tr>
        <tr><td>Caron et al. [15]</td><td>Clustering (1.3M)</td><td>45.1</td></tr>
        <tr><td>Feng et al. [16]</td><td>Feature Decoupling (1.3M)</td><td>45.3</td></tr>
        </tbody>
    </table>
    <h4>2. Pascal VOC 2012 Semantic Segmentation (ResNet-50)</h4>
    <table class="table table-condensed">
        <th>Method (ResNet-50)</th><th>Supervision (data amount)</th><th>% mIoU</th>
        <tbody>
        <tr><td>Krizhevsky et al. [1]</td><td>ImageNet labels (1.2M)</td><td>69.0</td></tr>
        <tr><td>Random</td><td>- (0)</td><td>42.4</td></tr>
        <tr><td>Walker et al. [12]*</td><td>Flow Prediction (1.26M)</td><td>54.5</td></tr>
        <tr><td>Pathak et al. [11]*</td><td>Motion Segmentation (1.6M)</td><td>54.6</td></tr>
        <tr><td>Ours</td><td>CMP (1.26M)</td><td>59.0</td></tr>
        </tbody>
    </table>
    <h4>3. COCO 2017 Instance Segmentation (ResNet-50)</h4>
    <table class="table table-condensed">
        <th>Method (ResNet-50)</th><th>Supervision (data amount)</th><th>Det. (% mAP)</th><th>Seg. (% mAP)</th>
        <tbody>
        <tr><td>Krizhevsky et al. [1]</td><td>ImageNet labels (1.2M)</td><td>37.2</td><td>34.1</td></tr>
        <tr><td>Random</td><td>- (0)</td><td>19.7</td><td>18.8</td></tr>
        <tr><td>Pathak et al. [11]*</td><td>Motion Segmentation (1.6M)</td><td>27.7</td><td>25.8</td></tr>
        <tr><td>Walker et al. [12]*</td><td>Flow Prediction (1.26M)</td><td>31.5</td><td>29.2</td></tr>
        <tr><td>Ours</td><td>CMP (1.26M)</td><td>32.3</td><td>29.8</td></tr>
        </tbody>
    </table>
    <h4>4. LIP Human Parsing (ResNet-50)</h4>
    <table class="table table-condensed">
        <th>Method (ResNet-50)</th><th>Supervision (data amount)</th><th>Single-Person (% mIoU)</th><th>Multi-Person (% mIoU)</th>
        <tbody>
        <tr><td>Krizhevsky et al. [1]</td><td>ImageNet labels (1.2M)</td><td>42.5</td><td>55.4</td></tr>
        <tr><td>Random</td><td>- (0)</td><td>32.5</td><td>35.0</td></tr>
        <tr><td>Pathak et al. [11]*</td><td>Motion Segmentation (1.6M)</td><td>36.6</td><td>50.9</td></tr>
        <tr><td>Walker et al. [12]*</td><td>Flow Prediction (1.26M)</td><td>36.7</td><td>52.5</td></tr>
        <tr><td>Ours</td><td>CMP (1.26M)</td><td>36.9</td><td>51.8</td></tr>
        <tr><td>Ours</td><td>CMP (4.57M)</td><td>40.2</td><td>52.9</td></tr>
        </tbody>
    </table>
    Note: Methods marked * have not reported the results in their paper, hence we reimplemented them to obtain the results.
    <br>
    <h4>References</h4>
    <ol>
        <li>Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In NeurIPS, 2012.</li>
        <li>Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, and Alexei A Efros. Context encoders: Feature learning by inpainting. In CVPR, 2016.</li>
        <li>Richard Zhang, Phillip Isola, and Alexei A Efros. Colorful image colorization. In ECCV. Springer, 2016.</li>
        <li>Richard Zhang, Phillip Isola, and Alexei A Efros. Split-brain autoencoders: Unsupervised learning by cross-channel prediction. In CVPR, 2017.</li>
        <li>Mehdi Noroozi, Hamed Pirsiavash, and Paolo Favaro. Representation learning by learning to count. In ICCV, 2017.</li>
        <li>Mehdi Noroozi and Paolo Favaro. Unsupervised learning of visual representations by solving jigsaw puzzles. In ECCV. Springer, 2016.</li>
        <li>Mehdi Noroozi, Ananth Vinjimoor, Paolo Favaro, and Hamed Pirsiavash. Boosting self-supervised learning via knowledge transfer. In CVPR, 2018.</li>
        <li>Simon Jenni and Paolo Favaro. Self-supervised feature learning by learning to spot artifacts. In CVPR, 2018.</li>
        <li>Gustav Larsson, Michael Maire, and Gregory Shakhnarovich. Colorization as a proxy task for visual understanding. In CVPR, 2017.</li>
        <li>Spyros Gidaris, Praveer Singh, and Nikos Komodakis. Unsupervised representation learning by predicting image rotations. In ICLR, 2018.</li>
        <li>Deepak Pathak, Ross B Girshick, Piotr Dollar, Trevor Darrell, and Bharath Hariharan. Learning features by watching objects move. In CVPR, 2017.</li>
        <li>Jacob Walker, Abhinav Gupta, and Martial Hebert. Dense optical flow prediction from a static image. In ICCV, 2015.</li>
        <li>T Nathan Mundhenk, Daniel Ho, and Barry Y Chen. Improvements to context based self-supervised learning. CVPR, 2018.</li>
        <li>A. Mahendran, J. Thewlis, and A. Vedaldi. Cross pixel optical flow similarity for self-supervised learning. In ACCV, 2018.</li>
        <li>Mathilde Caron, Piotr Bojanowski, Armand Joulin, and Matthijs Douze. Deep clustering for unsupervised learning of visual features. In ECCV, 2018.</li>
        <li>Zeyu Feng, Chang Xu, and Dacheng Tao. Self-Supervised Representation Learning by Rotation Feature Decoupling. In CVPR, 2019.</li>
    </ol>

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
