# CountSE: Soft Exemplar Open-set Object Counting

Shuai Liu, Peng Zhang, Shiwei Zhang, Wei Ke*

Official PyTorch implementation for [CountSE](https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_CountSE_Soft_Exemplar_Open-set_Object_Counting_ICCV_2025_paper.pdf).

## CountSE Architecture

<img src=img/architecture.jpg width="100%"/>

## Contents
* [📦 Preparation](#preparation)
* [🎯 Evaluation](#evaluation)
* [🚀 Training](#training)
* [📜 Citation](#citation)
* [🙏 Acknowledgements](#acknowledgements)

## Preparation
### 1. Clone Repository

```
git clone https://github.com/pppppz22/CountSE.git
```

### 2. Dataset Preparation

We use [FSC-147](https://github.com/cvlab-stonybrook/LearningToCountEverything) for training. Update the paths in `config/datasets_fsc147_val.json` for your installation. Evaluation data preparation for all supported datasets is described in the [evaluation guide](cross_dataset/README.md).

### 3. Set Up Environment
If you have already run CountGD, you can directly use its environment to run our code.

Install GCC and virtual environment. We used [Anaconda version 2024.02-1](https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh).

```
sudo apt update
sudo apt install build-essential

conda create -n countse python=3.9.19
conda activate countse
cd CountSE
pip install -r requirements.txt
export CC=/usr/bin/gcc-11 # this ensures that gcc 11 is being used for compilation
cd models/GroundingDINO/ops
python setup.py build install
python test.py # should result in 6 lines of * True
cd ../../../
```

If you have problems during the installation environment, you can check the [CountGD](https://github.com/niki-amini-naieni/CountGD/issues) or [Open-GrundingDINO](https://github.com/longzw1997/Open-GroundingDino/issues) repository issues for assistance.

### 4. Download Pre-Trained Weights

Download pre-training groudingdino weights 

  ```
  wget -P pretrained_ckpts https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
  ```

Download Bert

  ```
  python download_bert.py
  ```

## Training
You can use the following command to train the model. The --output_dir argument specifies the path where the model will be saved. If you need to resume training, you can use the --start_epoch parameter, which defaults to 0.
```
python -u main.py --output_dir ./countse_ckpt -c config/cfg_fsc147_val.py --datasets config/datasets_fsc147_val.json --pretrain_model_path ./pretrained_ckpts/groundingdino_swinb_cogcoor.pth --gpuid 0 --options text_encoder_type=./checkpoints/bert-base-uncased

```

## Evaluation

Download the trained [CountSE checkpoint](https://drive.google.com/file/d/1wJaWcrQB_z4LaQwApNChg_xN3DWvqfgP/view) and save it as `pretrained_ckpts/checkpoint_best_regular.pth`.

See the [evaluation guide](cross_dataset/README.md) for FSC-147 validation/test and CARPK/ShanghaiTech-A cross-dataset evaluation, including data preparation, inference settings, and commands.

**ShanghaiTech Part A — RMSE:** [![Papers with Code — ShanghaiTech Part A — RMSE](https://paperswithcode.co/api/v1/papers/98356/leaderboard-badge.svg?eval=10581&live=1)](https://paperswithcode.co/api/v1/papers/98356/leaderboard-badge-link?eval=10581)

**ShanghaiTech Part A — MAE:** [![Papers with Code — ShanghaiTech Part A — MAE](https://paperswithcode.co/api/v1/papers/98356/leaderboard-badge.svg?eval=10580&live=1)](https://paperswithcode.co/api/v1/papers/98356/leaderboard-badge-link?eval=10580)

**FSC-147 — MAE:** [![Papers with Code — FSC-147 — MAE](https://paperswithcode.co/api/v1/papers/98356/leaderboard-badge.svg?eval=10578&live=1)](https://paperswithcode.co/api/v1/papers/98356/leaderboard-badge-link?eval=10578)

## Citation

```
@inproceedings{liu2025countse,
  title={CountSE: Soft Exemplar Open-set Object Counting},
  author={Liu, Shuai and Zhang, Peng and Zhang, Shiwei and Ke, Wei},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={21536--21546},
  year={2025}
}
```

### Acknowledgements
Our code is based on [CountGD](https://github.com/niki-amini-naieni/CountGD). Please use [GitHub Issues](https://github.com/pppppz22/CountSE/issues) for questions.
