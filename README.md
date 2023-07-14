# SPL

The Official PyTorch code for “[A Weakly Supervised Semantic Segmentation Method based on Local Superpixel Transformation](https://europepmc.org/article/PPR/PPR635314)”.

![p2.png](SPL.png)

Our code will be made publicly available after the paper is accepted.

# Installation

Use the following command to prepare your environment.

```jsx
pip install -r requirements.txt
```

# **Execution**

## **Dataset & pretrained model**

- PASCAL VOC 2012
    - [Images](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
    - [Saliency maps](https://drive.google.com/file/d/19AjSmgdMlIZH4FXVZ5zjlUZcoZZCkwrI/view?usp=sharing) using Poolnet
- Pretrained models
    - [ImageNet-pretrained Model](https://drive.google.com/file/d/1WvSkPfAtfPzyxcgG58a1RlRayMYb3FBc/view?usp=share_link) for [ResNet38](https://arxiv.org/abs/1611.10080)

### Segmentation network

- We utilize [DeepLab-V2](https://arxiv.org/abs/1606.00915) for the segmentation network.
- Please see [deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch) for the implementation in PyTorch.

# Citation

If you use our codes and models in your research, please cite:

```jsx
@misc {PPR:PPR635314,
	Title = {A Weakly Supervised Semantic Segmentation Method based on Local Superpixel Transformation},
	Author = {Ma, Zhiming and Chen, Dali and Mo, Yilin and Chen, Yue and Zhang, Yuming},
	DOI = {10.21203/rs.3.rs-2714436/v1},
	Abstract = {Weakly supervised semantic segmentation (WSSS) can obtain pseudo-semantic masks through a weaker level of supervised labels, reducing the need for costly pixel-level annotations. However, the general class activation map (CAM)-based pseudo-mask acquisition method suffers from sparse coverage, leading to false positive and false negative regions that reduce accuracy. We propose a WSSS method based on local superpixel transformation that combines superpixel theory and image local information. Our method uses a superpixel local consistency weighted cross-entropy loss to correct erroneous regions and a post-processing method based on the adjacent superpixel affinity matrix (ASAM) to expand false negatives, suppress false positives, and optimize semantic boundaries. Our method achieves 73.4% mIoU on the PASCAL VOC 2012 validation set and 73.9% on the test set, and the ASAM post-processing method is validated on several state-of-the-art methods. If our paper is accepted, our code will be published.},
	Publisher = {Research Square},
	Year = {2023},
	URL = {https://doi.org/10.21203/rs.3.rs-2714436/v1},
}
```