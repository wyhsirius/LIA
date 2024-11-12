## Latent Image Animator: Learning to Animate Images via Latent Space Navigation (ICLR 2022) & LIA: Latent Image Animator (TPAMI 2024)
Yaohui Wang, Di Yang, François Brémond, Antitza Dantcheva
### [Project Page](https://wyhsirius.github.io/LIA-project/) | [Paper](https://openreview.net/pdf?id=7r6kDq0mK_)
This is the official PyTorch implementation of the ICLR 2022 paper "Latent Image Animator: Learning to Animate Images via Latent Space Navigation" and TPAMI 2024 paper "LIA: Latent Image Animator".

[![Replicate](https://replicate.com/wyhsirius/lia/badge)](https://replicate.com/wyhsirius/lia)

<img src="LIA.gif" width="500">

<a href="https://www.inria.fr/"><img height="80" src="assets/logo_inria.png"> </a>
<a href="https://univ-cotedazur.eu/"><img height="80" src="assets/logo_uca.png"> </a>

Abstract: *Due to the remarkable progress of deep generative models, animating images has become increasingly efficient, whereas associated results have become increasingly realistic. Current animation-approaches commonly exploit structure representation extracted from driving videos. Such structure representation is instrumental in transferring motion from driving videos to still images. However, such approaches fail in case the source image and driving video encompass large appearance variation. Moreover, the extraction of structure information requires additional modules that endow the animation-model with increased complexity. Deviating from such models, we here introduce the Latent Image Animator (LIA), a self-supervised autoencoder that evades need for structure representation. LIA is streamlined to animate images by linear navigation in the latent space. Specifically, motion in generated video is constructed by linear displacement of codes in the latent space. Towards this, we learn a set of orthogonal motion directions simultaneously, and use their linear combination, in order to represent any displacement in the latent space. Extensive quantitative and qualitative analysis suggests that our model systematically and significantly outperforms state-of-art methods on VoxCeleb, Taichi and TED-talk datasets w.r.t. generated quality.*

## BibTex
```bibtex
@inproceedings{
wang2022latent,
title={Latent Image Animator: Learning to Animate Images via Latent Space Navigation},
author={Yaohui Wang and Di Yang and Francois Bremond and Antitza Dantcheva},
booktitle={International Conference on Learning Representations},
year={2022}
}

@ARTICLE{10645735,
  author={Wang, Yaohui and Yang, Di and Bremond, Francois and Dantcheva, Antitza},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={LIA: Latent Image Animator}, 
  year={2024},
  pages={1-16},
}
```

## Requirements
- Python 3.7
- PyTorch 1.5+
- tensorboard
- moviepy
- av
- tqdm
- lpips

## 1. Animation demo

Download pre-trained checkpoints from [here](https://drive.google.com/drive/folders/1N4QcnqUQwKUZivFV-YeBuPyH4pGJHooc?usp=sharing) and put models under `./checkpoints`. We have provided several demo source images and driving videos in `./data`. 
To obtain demos, you could run following commands, generated results will be saved under `./res`.
```shell script
python run_demo.py --model vox --source_path ./data/vox/macron.png --driving_path ./data/vox/driving1.mp4 # using vox model
python run_demo.py --model taichi --source_path ./data/taichi/subject1.png --driving_path ./data/taichi/driving1.mp4 # using taichi model
python run_demo.py --model ted --source_path ./data/ted/subject1.png --driving_path ./data/ted/driving1.mp4 # using ted model
```
If you would like to use your own image and video, indicate `<SOURCE_PATH>` (source image), `<DRIVING_PATH>` (driving video), `<DATASET>` and run   
```shell script
python run_demo.py --model <DATASET> --source_path <SOURCE_PATH> --driving_path <DRIVING_PATH>
```
## 2. Datasets

Please follow the instructions in [FOMM](https://github.com/AliaksandrSiarohin/first-order-model) and [MRAA](https://github.com/snap-research/articulated-animation) to download and preprocess VoxCeleb, Taichi and Ted datasets. Put datasets under `./datasets` and organize them as follows:

#### Vox (Taichi, Ted)
```
Video Dataset (vox, taichi, ted)
|-- train
    |-- video1
        |-- frame1.png
        |-- frame2.png
        |-- ...
    |-- video2
        |-- frame1.png
        |-- frame2.png
        |-- ...
    |-- ...
|-- test
    |-- video1
        |-- frame1.png
        |-- frame2.png
        |-- ...
    |-- video2
        |-- frame1.png
        |-- frame2.png
        |-- ...
    |-- ...
```
## 3. Training
By default, we use `DistributedDataParallel` on 8 V100 for all datasets. To train the netowrk, run
```shell script
python train.py --dataset <DATSET> --exp_path <EXP_PATH> --exp_name <EXP_NAME>
```
The dataset list is as follows, `<DATASET>`: {`vox`,`taichi`,`ted`}. Tensorboard log and checkpoints will be saved in `<EXP_PATH>/<EXP_NAME>/log` and `<EXP_PATH>/<EXP_NAME>/chekcpoints` respectively.

To train from a checkpoint, run
```shell script
python train.py --dataset <DATASET> --exp_path <EXP_PATH> --exp_name <EXP_NAME> --resume_ckpt <CHECKPOINT_PATH>
```
## 4. Evaluation
To obtain reconstruction and LPIPS results, put checkpoints under `./checkpoints` and run
```shell script
python evaluation.py --dataset <DATASET> --save_path <SAVE_PATH>
```
Generated videos will be save under `<SAVE_PATH>`. For other evaluation metrics, we use the code from [here](https://github.com/AliaksandrSiarohin/pose-evaluation).
## 5. Linear manipulation
To obtain linear manipulation results of a single image, run
```shell script
python linear_manipulation.py --model <DATAET> --img_path <IMAGE_PATH> --save_folder <RESULTS_PATH>
```
By default, results will be saved under `./res_manipulation`.

## Acknowledgement
Part of the code is adapted from [FOMM](https://github.com/AliaksandrSiarohin/first-order-model) and [MRAA](https://github.com/snap-research/articulated-animation). We thank authors for their contribution to the community.
