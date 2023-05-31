# iWarpGAN
iWarpGAN: Disentangling Identity and Style to Generate Synthetic Iris Images
Generative Adversarial Networks (GANs) have shown success in approximating complex distributions for synthetic image generation and for editing specific portions of an input image, particularly in faces. However, current GAN-based methods for generating biometric images, such as iris, have limitations in controlling the identity of the generated images, i.e., the synthetically generated images often closely resemble images in the training dataset. Further, the generated images often lack diversity in terms of the number of unique identities represented in them. To overcome these issues, we propose iWarpGAN that disentangles identity and style in the context of the iris modality by using two transformation pathways: Identity Transformation Pathway to generate unique identities from the training set, and Style Transformation Pathway to extract the style code from a reference image and output an iris image using this style. By concatenating the transformed identity code and reference style code, iWarpGAN generates iris images with both inter and intra-class variations. The efficacy of the proposed method in generating Iris DeepFakes is evaluated both qualitatively and quantitatively using ISO/IEC 29794-6 Standard Quality Metrics and the VeriEye iris matcher. Finally, the utility of the synthetically generated images is demonstrated by improving the performance of multiple deep learning based iris matchers that augment synthetic data with real data during the training process.


Requirements
Linux and Windows are supported, but we recommend Linux for performance and compatibility reasons.
1–8 high-end NVIDIA GPUs with at least 12 GB of memory.
CUDA toolkit 11.1 or later.
GCC 7 or later (Linux) or Visual Studio (Windows) compilers.
Python libraries: see environment.yml for exact library dependencies. You can use the following commands with Miniconda3 to create and activate your StyleGAN3 Python environment:
- conda env create -f environment.yml
- conda activate warp

Getting started

Preparing datasets
Custom datasets can be created from a folder containing images; see python dataset_tool.py --help for more information.
python dataset_tool.py --source=./image-folder/ --dest=./prc_folder

Training

Training is done using train.py. For example:
python train.py --outdir=./outdir/ --cfg=stylegan3-t --data=./prc_folder --gpus=n-gpus --batch=n-batch --gamma=8.2 --mirror=1 --cond True

