# CountSE evaluation

Evaluate FSC-147 and run cross-dataset evaluation on CARPK and ShanghaiTech-A using the FSC-147-trained [CountSE checkpoint](https://drive.google.com/file/d/1wJaWcrQB_z4LaQwApNChg_xN3DWvqfgP/view) directly, without target-dataset fine-tuning. This folder reuses `main_inference.py` in the repository root.

## 1. Download datasets

- [FSC-147](https://github.com/cvlab-stonybrook/LearningToCountEverything): prepare the images and CountSE-format annotations, then update the paths in `config/datasets_fsc147_val.json` and `config/datasets_fsc147_test.json`.
- [CARPK](https://lafi.github.io/LPN/): download the dataset and follow its access terms. Reference: Hsieh et al., *Drone-based Object Counting by Spatially Regularized Regional Proposal Networks*, ICCV 2017.
- [ShanghaiTech Part A](https://github.com/desenzhou/ShanghaiTechDataset): download the dataset. Reference: Zhang et al., *Single-Image Crowd Counting via Multi-Column Convolutional Neural Network*, CVPR 2016.

Keep the original images and annotations in this layout:

```text
data/
├── CARPK_devkit/data/
│   ├── Images/*.png
│   ├── Annotations/*.txt
│   └── ImageSets/test.txt
└── ShanghaiTech/part_A/test_data/
    ├── images/IMG_*.jpg
    └── ground_truth/GT_IMG_*.mat
```

## 2. Prepare annotations

For CARPK and ShanghaiTech-A, run from the repository root:

```bash
python cross_dataset/prepare_cross_dataset.py carpk \
  --dataset-root data/CARPK_devkit --output-dir data/carpk
python cross_dataset/prepare_cross_dataset.py sha \
  --dataset-root data/ShanghaiTech/part_A --output-dir data/sha
```

The script converts the official test annotations to COCO format and generates `dataset.json`. All original vehicle boxes or head points are retained; density maps are not required.

## 3. Run evaluation

Use the CountSE environment and weights described in the main README. Set `--annotation_mode fsc147` for FSC-147 (exclude the three appended exemplar boxes), or `--annotation_mode no_exemplar` for CARPK and ShanghaiTech-A (retain all GT).

For CARPK and ShanghaiTech-A, prompts are `car` and `people`, respectively. Both evaluations use box threshold 0.23, text threshold 0, batch size 1, FP32, seed 215, and SEG (budget 18, top 15 candidates per scale). Images are resized with aspect ratio preserved: short side 800, maximum long side 1333. As in the verified rerun, cropping and extra test-time normalization are disabled; no manual visual exemplars are used.

**CARPK and ShanghaiTech-A:**

```bash
python -u main_inference.py --eval --output_dir outputs/carpk \
  -c cross_dataset/cfg_carpk_test.py --datasets data/carpk/dataset.json \
  --pretrain_model_path pretrained_ckpts/checkpoint_best_regular.pth \
  --gpuid 0 --num_exemplars 0 --annotation_mode no_exemplar --seed 215 \
  --options text_encoder_type=checkpoints/bert-base-uncased

python -u main_inference.py --eval --output_dir outputs/sha \
  -c cross_dataset/cfg_sha_test.py --datasets data/sha/dataset.json \
  --pretrain_model_path pretrained_ckpts/checkpoint_best_regular.pth \
  --gpuid 0 --num_exemplars 0 --annotation_mode no_exemplar --seed 215 \
  --options text_encoder_type=checkpoints/bert-base-uncased
```

**FSC-147 validation and test:**

```bash
python -u main_inference.py --eval --output_dir outputs/fsc147_val \
  -c config/cfg_fsc147_val.py --datasets config/datasets_fsc147_val.json \
  --pretrain_model_path pretrained_ckpts/checkpoint_best_regular.pth \
  --gpuid 0 --annotation_mode fsc147 --seed 215 \
  --options text_encoder_type=checkpoints/bert-base-uncased \
  --crop --remove_bad_exemplar

python -u main_inference.py --eval --output_dir outputs/fsc147_test \
  -c config/cfg_fsc147_test.py --datasets config/datasets_fsc147_test.json \
  --pretrain_model_path pretrained_ckpts/checkpoint_best_regular.pth \
  --gpuid 0 --annotation_mode fsc147 --seed 215 \
  --options text_encoder_type=checkpoints/bert-base-uncased \
  --crop --remove_bad_exemplar
```
