## iWarpGAN: Disentangling Identity and Style to Generate Synthetic Iris Images<br><sub>Official PyTorch implementation</sub>


**iWarpGAN: Disentangling Identity and Style to Generate Synthetic Iris Images**<br>
Shivangi Yadav and Arun Ross<br>
https://arxiv.org/abs/2305.12596<br>

Abstract: *Generative Adversarial Networks (GANs) have shown success in approximating complex distributions for synthetic image generation and for editing specific portions of an input image, particularly in faces. However, current GAN-based methods for generating biometric images, such as iris, have limitations in controlling the identity of the generated images, i.e., the synthetically generated images often closely resemble images in the training dataset. Further, the generated images often lack diversity in terms of the number of unique identities represented in them. To overcome these issues, we propose iWarpGAN that disentangles identity and style in the context of the iris modality by using two transformation pathways: Identity Transformation Pathway to generate unique identities from the training set, and Style Transformation Pathway to extract the style code from a reference image and output an iris image using this style. By concatenating the transformed identity code and reference style code, iWarpGAN generates iris images with both inter and intra-class variations. The efficacy of the proposed method in generating Iris DeepFakes is evaluated both qualitatively and quantitatively using ISO/IEC 29794-6 Standard Quality Metrics and the VeriEye iris matcher. Finally, the utility of the synthetically generated images is demonstrated by improving the performance of multiple deep learning based iris matchers that augment synthetic data with real data during the training process.*


## Release notes

This repository is a Pytorch implementation of iWarpGAN that utilizes features from https://github.com/chi0tzp/WarpedGANSpace/blob/master/train.py and https://github.com/NVlabs/stylegan3#readme


## Requirements

* Linux and Windows are supported, but we recommend Linux for performance and compatibility reasons.
* 1&ndash;8 high-end NVIDIA GPUs with at least 12 GB of memory. We have done all testing and development using Tesla V100 and A100 GPUs.
* 64-bit Python 3.8 and PyTorch 1.9.0 (or later). See https://pytorch.org for PyTorch install instructions.
* CUDA toolkit 11.1 or later.  (Why is a separate CUDA toolkit installation required?  See [Troubleshooting](./docs/troubleshooting.md#why-is-cuda-toolkit-installation-necessary)).
* GCC 7 or later (Linux) or Visual Studio (Windows) compilers.  Recommended GCC version depends on CUDA version, see for example [CUDA 11.4 system requirements](https://docs.nvidia.com/cuda/archive/11.4.1/cuda-installation-guide-linux/index.html#system-requirements).
* Python libraries: see [environment.yml](./environment.yml) for exact library dependencies.  You can use the following commands with Miniconda3 to create and activate your StyleGAN3 Python environment:
  - `conda env create -f environment.yml`
  - `conda activate warp`

The code relies heavily on custom PyTorch extensions that are compiled on the fly using NVCC. On Windows, the compilation requires Microsoft Visual Studio. We recommend installing [Visual Studio Community Edition](https://visualstudio.microsoft.com/vs/) and adding it into `PATH` using `"C:\Program Files (x86)\Microsoft Visual Studio\<VERSION>\Community\VC\Auxiliary\Build\vcvars64.bat"`.


## Getting started

Pre-trained networks are stored as `*.pkl` files that can be referenced using local filenames or URLs:

```.bash
# Generate an image using pre-trained model (General GAN model to generate images using noise as input)
python gen_images.py --outdir=out --data=test_data --trunc=1 --seeds=2 \
    --network=network.pkl

# Generate an image using pre-trained model (Image translative GAN model to generate images using image an input)
python gen_images.py --outdir=out --data=test_data --use_es=True --use_ed=True --trunc=1 --seeds=2 \
    --network=network.pkl
```

## Preparing datasets

Datasets are stored in a folder with a metadata file `dataset.json` for labels. Custom datasets can be created from a folder containing images; see [`python dataset_tool.py --help`] for more information:

```.bash
python dataset_tool.py --source=./image-folder/ --dest=./prc_folder
```

## Training

You can train new networks using `train.py`. For example:

```.bash
# General GAN model to generate images using noise as input
python train.py --outdir=./outdir/ --cfg=stylegan3-t --data=./prc_folder --gpus=n-gpus --batch=n-batch --gamma=8.2 --mirror=1 --cond True 

#Image translative GAN model to generate images using image an input
python train.py --outdir=./outdir/ --cfg=stylegan3-t --data=./prc_folder --gpus=n-gpus --batch=n-batch --gamma=8.2 --mirror=1 --cond True --use_es=True  --use_ed==True

```

For Image translative model, the best results are obtained when Style Transformation Pathway is trained first for atleast 200 ticks (set use_es = True and use_ed=False) and then style and identity transformation pathway are trained alternatively (both of them True).


## Quality metrics

`train.py` computes FID for each network during training. When not required, this computation can be disabled with `--metrics=none` to speed up the training process.

Additional quality metrics can also be computed after the training. Check metrics for that.


References:
1. [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500), Heusel et al. 2017
2. [Demystifying MMD GANs](https://arxiv.org/abs/1801.01401), Bi&nacute;kowski et al. 2018
3. [Improved Precision and Recall Metric for Assessing Generative Models](https://arxiv.org/abs/1904.06991), Kynk&auml;&auml;nniemi et al. 2019
4. [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948), Karras et al. 2018
5. [Alias-Free Generative Adversarial Networks](https://nvlabs.github.io/stylegan3), Karras et al. 2021
6. [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498), Salimans et al. 2016
7. [WarpedGANSpace: Finding non-linear RBF paths in GAN latent space](https://github.com/chi0tzp/WarpedGANSpace), Tzelepis et al. 2021

## Citation

```
@article{yadav2023,
  title={{iWarpGAN: Disentangling Identity and Style to Generate Synthetic Iris Images}},
  author={Yadav, Shivangi and Ross, Arun},
  journal={arXiv preprint arXiv:2305.12596},
  year={2023}
}
```

## Development

This is code is under development to improve the network and it's performance

