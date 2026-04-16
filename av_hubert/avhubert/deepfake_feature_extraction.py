# This file incorporates code from Reiss et al. FACTOR (https://github.com/talreiss/FACTOR)

# Example usage:
# python -m avhubert.deepfake_feature_extraction --dataset FakeAVCeleb --metadata /path/to/FakeAVCeleb/meta_data.csv --ckpt_path /path/to/self_large_vox_433h.pt --data_path /path/to/FakeAVCeleb/output_videos_and_audio --save_path /path/to/FakeAVCeleb/output_videos_and_audio/features --category all
# python -m avhubert.deepfake_feature_extraction --dataset AV1M --metadata /path/to/AV1M/meta_data.csv --ckpt_path /path/to/self_large_vox_433h.pt --data_path /path/to/AV1M/output_videos_and_audio  --save_path /path/to/AV1M/output_videos_and_audio/features --category all --trimmed

import argparse
import os
import csv

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from python_speech_features import logfbank
from tqdm import tqdm
from fairseq import checkpoint_utils

# Fix deprecation in numpy
np.float = np.float64
np.int = np.int_

from . import hubert_pretraining, hubert, hubert_asr
from . import avhubert_utils

FPS = 25

def load_model(ckpt_path):
    models, _, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    model = models[0]
    if hasattr(model, "decoder"):
        print("Checkpoint: fine-tuned")
        model = model.encoder.w2v_model
    else:
        print("Checkpoint: pre-trained w/o fine-tuning")
    model.cuda().eval()
    return model, task

def load_transforms(task):
    return avhubert_utils.Compose([
        avhubert_utils.Normalize(0.0, 255.0),
        avhubert_utils.CenterCrop((task.cfg.image_crop_size, task.cfg.image_crop_size)),
        avhubert_utils.Normalize(task.cfg.image_mean, task.cfg.image_std)
    ])

def compute_starting_silence(audio_path, threshold=0.0005, sr=16000):
    # compute the starting silence in seconds
    audio, _ = librosa.load(audio_path, sr=sr)
    for i, sample in enumerate(audio):
        if abs(sample) > threshold:
            return i / sr
    return len(audio) / sr

def load_audio(path, silence_duration=0, sample_rate=16000, stack_order_audio=4):
    wav_data, sr = librosa.load(path, sr=sample_rate)
    assert sr == sample_rate and len(wav_data.shape) == 1

    skiped_frames = int(silence_duration * FPS) * 640
    if silence_duration > 0:
        skiped_frames += 640
    wav_data = wav_data[skiped_frames:]

    audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32)

    if len(audio_feats) % stack_order_audio != 0:
        pad = stack_order_audio - len(audio_feats) % stack_order_audio
        audio_feats = np.concatenate([audio_feats, np.zeros((pad, audio_feats.shape[1]), dtype=audio_feats.dtype)])

    audio_feats = audio_feats.reshape(-1, stack_order_audio * audio_feats.shape[1])
    audio_feats = torch.from_numpy(audio_feats.astype(np.float32))
    with torch.no_grad():
        audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])
    return audio_feats

def extract_features(model, video_path, audio_path, transform, trimmed):
    frames = avhubert_utils.load_video(video_path)
    frames = transform(frames)
    frames = torch.FloatTensor(frames).unsqueeze(0).unsqueeze(0).cuda()

    audio_silence = compute_starting_silence(audio_path) if trimmed else 0
    audio = load_audio(audio_path, silence_duration=audio_silence)[None, :, :].transpose(1, 2).cuda()

    skip_frames = int(audio_silence * FPS) + 1 if audio_silence > 0 else 0
    print(f"Trimmed: {trimmed}, Skipped frames: {skip_frames}")
    frames = frames[:, :, skip_frames:]

    min_len = min(frames.shape[2], audio.shape[-1])
    frames, audio = frames[:, :, :min_len], audio[:, :, :min_len]

    with torch.no_grad():
        f_audio, _ = model.extract_finetune({"video": None, "audio": audio}, None, None)
        f_video, _ = model.extract_finetune({"video": frames, "audio": None}, None, None)
        f_mm, _ = model.extract_finetune({"video": frames, "audio": audio}, None, None)

    return f_audio.squeeze(0).cpu().numpy(), f_video.squeeze(0).cpu().numpy(), f_mm.squeeze(0).cpu().numpy()

def process_av1m(args, model, transform):
    file_paths = set()
    with open(args.metadata, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            file_paths.add(row["path"])

    for _, file_path in enumerate(tqdm(file_paths)):
        mouth_roi_path = os.path.join(args.data_path, file_path[:-4] + "_roi.mp4")
        audio_path = os.path.join(args.data_path, file_path[:-4] + ".wav")

        save_path = os.path.join(args.save_path, file_path.replace(".mp4", ".npz"))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if (not args.overwrite) and os.path.exists(save_path):
            print(f"Skipping (exists): {save_path}")
            continue

        try:
            feature_audio, feature_vid, feature_multimodal = extract_features(model, mouth_roi_path, audio_path, transform, args.trimmed)
        except Exception as e:
            print(f"Unprocessed for file: {mouth_roi_path}; error {e}")
            continue

        save_dict = {
            "visual": feature_vid,
            "audio": feature_audio,
            "multimodal": feature_multimodal,
        }

        np.savez(save_path, **save_dict)

def process_fakeavceleb(args, model, transform, category):
    file_paths = set()

    with open(args.metadata, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["type"] == category:
                path = os.path.join(row["dir"].replace("FakeAVCeleb/", ""), row["path"])
                file_paths.add(path)

    for _, file_path in enumerate(tqdm(file_paths)):
        mouth_roi_path = os.path.join(args.data_path, file_path[:-4] + "_roi.mp4")
        audio_path = os.path.join(args.data_path, file_path[:-4] + ".wav")

        save_path = os.path.join(args.save_path, file_path.replace(".mp4", ".npz"))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if (not args.overwrite) and os.path.exists(save_path):
            print(f"Skipping (exists): {save_path}")
            continue

        try:
            feature_audio, feature_vid, feature_multimodal = extract_features(model, mouth_roi_path, audio_path, transform, args.trimmed)
        except Exception as e:
            print(f"Unprocessed for file: {mouth_roi_path}; error {e}")
            continue

        save_dict = {
            "visual": feature_vid,
            "audio": feature_audio,
            "multimodal": feature_multimodal,
        }

        np.savez(save_path, **save_dict)

def process_avlips(args, model, transform):
    file_paths = set()

    # Assume AVLips dataset contains 0_real and 1_fake subdirectories
    subsets = ["0_real", "1_fake"]
    for sub in subsets:
        input_dir = os.path.join(args.data_path, sub)
        if not os.path.isdir(input_dir):
            print(f"[WARN] AVLips: Input directory not found: {input_dir}")
            continue

        for fname in os.listdir(input_dir):
            if not fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                continue
            file_paths.add(os.path.join(sub, fname)) 

    for file_path in tqdm(file_paths):
        mouth_roi_path = os.path.join(args.data_path, file_path)
        audio_path = os.path.join(args.data_path, file_path[:-8] + ".wav")

        save_path = os.path.join(args.save_path, file_path.replace(".mp4", ".npz"))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if not args.overwrite and os.path.exists(save_path):
            print(f"Skipping (exists): {save_path}")
            continue

        try:
            feature_audio, feature_vid, feature_multimodal = extract_features(model, mouth_roi_path, audio_path, transform, args.trimmed)
        except Exception as e:
            print(f"Unprocessed for file: {mouth_roi_path}; error: {e}")
            continue

        save_dict = {
            "visual": feature_vid,
            "audio": feature_audio,
            "multimodal": feature_multimodal,
        }

        np.savez(save_path, **save_dict)


def main():
    parser = argparse.ArgumentParser(description="Extract AVHubert features")
    parser.add_argument("--dataset", type=str, default="AV1M", help="Dataset to extract features for")
    parser.add_argument("--metadata", type=str, default="av1m_metadata/train_metadata.csv", help="Path to the dataset metadata")
    parser.add_argument("--split", default="train", help="For AV1M: data split to process (e.g., val, train)")
    parser.add_argument("--ckpt_path", type=str, default="self_large_vox_433h.pt", help="Path to AVHubert checkpoint")
    parser.add_argument("--data_path", type=str, default="av1m_preprocessed/", help="Path to the root folder of preprocessed data")
    parser.add_argument("--save_path", type=str, default="av1m_features/", help="Output directory for saving features")
    parser.add_argument("--category", choices=["RealVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-RealAudio", "FakeVideo-FakeAudio", "all"], default="all", help="For FakeAVCeleb: select category")
    parser.add_argument("--trimmed", action="store_true", help="Whether to trim to starting silence or not")
    parser.add_argument("--overwrite", action="store_true", help="If set, recompute and overwrite existing .npz features")

    args = parser.parse_args()

    model, task = load_model(args.ckpt_path)
    transform = load_transforms(task)

    if args.dataset == "AV1M":
        process_av1m(args, model, transform)
        
    elif args.dataset == "FakeAVCeleb":
        if args.category == "all":
            categories = ["RealVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]
        elif args.category:
            categories = [args.category]

        for category in categories:
            process_fakeavceleb(args, model, transform, category)
            
    elif args.dataset == "AVLips":
        process_avlips(args, model, transform)

if __name__ == "__main__":
    main()
