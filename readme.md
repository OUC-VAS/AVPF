<div align="center">

# AVPF: Audio-Visual Pseudo-Fakes for Deepfake Detection

[![Paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2604.09110)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB.svg)](https://www.python.org/)

Official PyTorch Implementation of AVPF

</div>

<br/>


## 📣 News
* **[2026.04]** Paper is submitted to [arXiv](https://arxiv.org/abs/2604.09110).

## ⏳ Todo

- [x] Release testing code.
- [x] Release testing weights.
- [ ] Release training code.

---

## 🏆 State-of-the-Art Performance

| Method | Venue | Train set | Used data | Cross eval. | Trim test | FAVC (AUC) | FAVC (AP) | AV1M (AUC) | AV1M (AP) | AVLips (AUC) | AVLips (AP) | Average (AUC) | Average (AP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AVAD | CVPR'23 | LRS | R | ✅ | ✅ | 84.7 | 99.5 | 54.3 | 76.3 | 73.2 | 77.1 | 70.7 | 84.3 |
| SpeechForensics | NeurIPS'24 | VoxCeleb2 | R | ✅ | ✅ | **98.8** | **100.0** | 68.2 | 83.5 | *92.4* | *94.9* | 86.5 | *92.8* |
| AVH-Align | CVPR'25 | VoxCeleb2 | R | ✅ | ✅ | 94.6 | 99.8 | *83.5* | *93.5* | 86.6 | 76.8 | *88.2* | 90.0 |
| **AVPF (Ours)** | **-** | **VoxCeleb2** | **R** | ✅ | ✅ | *97.8* | *99.9* | **89.2** | **96.2** | **97.6** | **97.8** | **94.9** | **98.0** |

## Resources

The pre-trained checkpoint, `AVPF_pretrain.ckpt`, is available for download on [Google Drive](https://drive.google.com/drive/folders/1JyytOlaYl3yKLNOODhxQkEzffPQ5tA72?usp=sharing).

Please download and place the weights in the `ckpt/` directory before running the evaluation scripts.

---

## ⚙️ Environment Setup
This repository has been tested with **Python 3.10**
## Set-up AV-Hubert
```bash 
cd av_hubert/fairseq
pip install --editable ./
cd ../avhubert
# install additional files for AV-Hubert
mkdir -p content/data/misc/
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -O content/data/misc/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d content/data/misc/shape_predictor_68_face_landmarks.dat.bz2
wget --content-disposition https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/raw/master/preprocessing/20words_mean_face.npy -O content/data/misc/20words_mean_face.npy
cd ../../

# download avhubert checkpoint
wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/self_large_vox_433h.pt
mv self_large_vox_433h.pt av_hubert/avhubert/self_large_vox_433h.pt
```

### 1. Core Deep Learning Frameworks
Due to the specific architectural requirements of the AV-HuBERT backbone, strictly adhere to the following exact versions:
* `torch == 2.2.0` (crucial for AV-HuBERT compatibility)
* `torchvision == 0.17.0`
* `lightning == 2.4.0`

### 2. Multimedia Processing
For processing high-fidelity audio and video data in datasets like AV1M and FakeAVCeleb:
* **Audio:** `librosa == 0.9.1`, `python_speech_features == 0.6`
* **Video/Image:** `opencv-python == 4.10.0.84`, `scikit-video == 1.1.11`, `pillow == 11.3.0`
* **System Level:** `ffmpeg >= 4.3` (Requires system installation: `sudo apt install ffmpeg`)

### 3. Face Alignment & Landmark Detection
* `dlib == 19.24.9` (Note: requires `cmake` to be installed on your system prior to pip installation)

### 4. Data Handling & Utilities
* `numpy == 1.26.4`
* `pandas == 2.1.1`
* `scikit-learn == 1.3.2`
* `tqdm == 4.65.2`
* `PyYAML == 6.0.2`

For your convenience, we have provided a `requirements.txt` file containing all the strictly required versions. You can easily install all Python dependencies by running:

```bash
pip install -r requirements.txt
```

## Feature extraction

### 1. Preprocess video files

Run `deepfake_preprocess.py` from `av_hubert/avhubert`.  
Supported datasets: `AV1M`, `FakeAVCeleb`, `AVLips`.

**AV-Deepfake1M**
```bash
python deepfake_preprocess.py \
    --dataset AV1M \
    --metadata /path/to/av1m_metadata/train_metadata.csv \
    --data_path /path/to/AV1M_root \
    --save_path /path/to/save/output_videos_and_audio
```
**FakeAVCeleb**
```
python deepfake_preprocess.py \
    --dataset FakeAVCeleb \
    --metadata /path/to/FakeAVCeleb_metadata.csv \
    --data_path /path/to/FakeAVCeleb_root \
    --save_path /path/to/save/output_videos_and_audio \
    --category all
```
    
**AVLips**
```
python deepfake_preprocess.py \
    --dataset AVLips \
    --data_path /path/to/AVLips_root \
    --save_path /path/to/save/output_videos_and_audio
```

Note: For AVLips, the script expects subdirectories 0_real/ and 1_fake/ under --data_path. The --metadata argument is not required.

### 2. Extract features
Run deepfake_feature_extraction.py from av_hubert/avhubert.
Add --trimmed to discard initial silence based on audio onset detection (recommended for videos with variable start silence).
Add --overwrite to recompute and overwrite existing .npz feature files.

**AV-Deepfake1M**

```
python deepfake_feature_extraction.py \
    --dataset AV1M \
    --metadata /path/to/av1m_metadata/train_metadata.csv \
    --ckpt_path self_large_vox_433h.pt \
    --data_path /path/to/preprocessed/data \
    --save_path /path/to/save/features \
    --trimmed
```

**FakeAVCeleb**

```
python deepfake_feature_extraction.py \
    --dataset FakeAVCeleb \
    --metadata /path/to/FakeAVCeleb_metadata.csv \
    --ckpt_path self_large_vox_433h.pt \
    --data_path /path/to/preprocessed/data \
    --save_path /path/to/save/features \
    --category all \
    --trimmed
```

**AVLips**

```
python deepfake_feature_extraction.py \
    --dataset AVLips \
    --ckpt_path self_large_vox_433h.pt \
    --data_path /path/to/preprocessed/data \
    --save_path /path/to/save/features \
    --trimmed
```
Note: For AVLips, --metadata is not needed. The script will automatically process videos found in 0_real/ and 1_fake/ subdirectories.
    
## Testing the Model

After training or obtaining a pretrained checkpoint, you can evaluate the model on a test set using `train_test.py` with the `--test` flag.

### 1. Prepare the configuration file

Modify the YAML configuration file (`configs/test_config.yaml`). 

### 2. Run the test script
Execute the following command from the project root:
```
python train_test.py --config_path configs/test_config.yaml --test
```

## Acknowledgements 
We would like to thank the contributors of [AVH-Align](https://github.com/bit-ml/AVH-Align), [AV-Hubert](https://github.com/facebookresearch/av_hubert), [SelfBlendedImages
](https://github.com/mapooon/SelfBlendedImages)and all related repositories, for their open research and contributions.

## Citation
If you find this useful, please cite our paper:
```
@article{wei2026generalizing,
  title={Generalizing Video DeepFake Detection by Self-generated Audio-Visual Pseudo-Fakes},
  author={Wei, Zihe and Li, Yuezun},
  journal={arXiv preprint arXiv:2604.09110},
  year={2026}
}
```
