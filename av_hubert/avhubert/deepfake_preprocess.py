# This file incorporates code from Reiss et al. FACTOR (https://github.com/talreiss/FACTOR)

# Example usage:
# python deepfake_preprocess.py --dataset FakeAVCeleb --metadata /path/to/FakeAVCeleb/meta_data.csv --data_path /path/to/FakeAVCeleb --save_path /path/to/output_videos_and_audio
# python deepfake_preprocess.py --dataset AV1M --metadata /path/to/AV1M/test_labels.csv --data_path /path/to/AV1M/test_videos --save_path /path/to/AV1M/test_output

import argparse
import os
import subprocess
import numpy as np
import cv2
import csv
import dlib
import skvideo.io
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed
from preparation.align_mouth import landmarks_interpolate, crop_patch, write_video_ffmpeg

# Backward compatibility of np.float and np.int
np.float = np.float64
np.int = np.int_

# Constants for both datasets
FACE_PREDICTOR_PATH = "content/data/misc/shape_predictor_68_face_landmarks.dat"
MEAN_FACE_PATH = "content/data/misc/20words_mean_face.npy"
STD_SIZE = (256, 256)
STABLE_PNTS_IDS = [33, 36, 39, 42, 45]

def is_valid_file(path, min_size=1024):
    return os.path.exists(path) and os.path.getsize(path) >= min_size

def detect_landmark(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)
    coords = None
    for (_, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def preprocess_video(input_video_dir, video_filename, output_video_dir, face_predictor_path, mean_face_path):
    os.makedirs(output_video_dir, exist_ok=True)

    base_name = os.path.splitext(video_filename)[0]
    roi_path = os.path.join(output_video_dir, base_name + '_roi.mp4')
    audio_fn = os.path.join(output_video_dir, base_name + '.wav')
    input_path = os.path.join(input_video_dir, video_filename)

    # Process successfully completed only if both ROI and audio files are valid
    if is_valid_file(roi_path) and is_valid_file(audio_fn):
        return True

    # Remove residual corrupted or empty files
    for path in [roi_path, audio_fn]:
        if os.path.exists(path) and os.path.getsize(path) == 0:
            print(f"[INFO] Removing empty file: {path}")
            os.remove(path)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_predictor_path)
    mean_face_landmarks = np.load(mean_face_path)

    try:
        videogen = skvideo.io.vread(input_path)
    except Exception as e:
        print(f"Failed to read video: {input_path}, error: {e}")
        return False

    frames = np.array([frame for frame in videogen])
    landmarks = [detect_landmark(frame, detector, predictor) for frame in frames]
    preprocessed_landmarks = landmarks_interpolate(landmarks)

    try:
        rois = crop_patch(
            input_path,
            preprocessed_landmarks,
            mean_face_landmarks,
            STABLE_PNTS_IDS,
            STD_SIZE,
            window_margin=12,
            start_idx=48,
            stop_idx=68,
            crop_height=96,
            crop_width=96
        )
    except Exception as e:
        print(f"Failed to preprocess video: {input_path}; passing whole video. error: {e}")
        rois = frames[..., ::-1]

    try:
        write_video_ffmpeg(rois, roi_path, "/usr/bin/ffmpeg")
    except Exception as e:
        print(f"Failed to write roi video: {roi_path}, error: {e}")
        if os.path.exists(roi_path):
            os.remove(roi_path)
        return False

    # Verify successful video generation
    if not is_valid_file(roi_path):
        print(f"Invalid roi file generated: {roi_path}")
        if os.path.exists(roi_path):
            os.remove(roi_path)
        return False

    result = subprocess.run([
        "/usr/bin/ffmpeg",
        "-i", input_path,
        "-f", "wav",
        "-vn",
        "-y", audio_fn,
        "-loglevel", "quiet"
    ])

    if result.returncode != 0 or not is_valid_file(audio_fn):
        print(f"Failed to extract audio: {audio_fn}")
        if os.path.exists(audio_fn):
            os.remove(audio_fn)
        if os.path.exists(roi_path):
            os.remove(roi_path)
        return False

    return True

def process_av1m(metadata_file_path, path_to_images_root, save_path, max_workers):
    with open(metadata_file_path, "r") as f, ProcessPoolExecutor(max_workers=max_workers) as executor:
        reader = csv.DictReader(f)

        futures = {
            executor.submit(
                preprocess_video,
                path_to_images_root,
                row['path'],
                save_path,
                FACE_PREDICTOR_PATH,
                MEAN_FACE_PATH
            ): (path_to_images_root, row['path'])
            for row in reader
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing... "):
            input_dir, filename = futures[future]
            try:
                result = future.result()
                if not result:
                    print(f"[WARN] Failed to process video: {os.path.join(input_dir, filename)}")
            except Exception as e:
                print(f"[ERROR] Error in video {os.path.join(input_dir, filename)}: {e}")


def process_fakeavceleb(category, metadata_file_path, input_root, save_path, max_workers):
    selected_videos = []
    with open(metadata_file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["type"] == category:
                original_file_path = row["dir"].replace("FakeAVCeleb/", "")
                filename = row["path"]
                input_dir = os.path.join(input_root, original_file_path)
                output_dir = os.path.join(save_path, original_file_path)
                selected_videos.append((input_dir, filename, output_dir))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                preprocess_video,
                input_dir,
                filename,
                output_dir,
                FACE_PREDICTOR_PATH,
                MEAN_FACE_PATH
            ): (input_dir, filename)
            for input_dir, filename, output_dir in selected_videos
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {category}..."):
            input_dir, filename = futures[future]
            try:
                result = future.result()
                if not result:
                    print(f"[WARN] Failed to process video: {os.path.join(input_dir, filename)}")
            except Exception as e:
                print(f"[ERROR] Error in video {os.path.join(input_dir, filename)}: {e}")

def process_avlips(input_root, save_path, max_workers):
    """
    AVLips dataset directory structure convention:
        root_path/
            0_real/
            1_fake/
    Assumes original video files (.mp4, etc.) are located in these subdirectories.
    Applies standard preprocessing and preserves the directory structure in save_path.
    """
    subsets = ["0_real", "1_fake"]
    selected_videos = []

    # Collect all videos to process
    for sub in subsets:
        input_dir = os.path.join(input_root, sub)
        output_dir = os.path.join(save_path, sub)

        if not os.path.isdir(input_dir):
            print(f"[WARN] AVLips: input dir not found: {input_dir}")
            continue

        for fname in os.listdir(input_dir):
            # Supported video extensions can be expanded here
            if not fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                continue
            selected_videos.append((input_dir, fname, output_dir))

    if not selected_videos:
        print(f"[WARN] No videos found under {input_root} for AVLips.")
        return

    # Parallel preprocessing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                preprocess_video,
                input_dir,
                filename,
                output_dir,
                FACE_PREDICTOR_PATH,
                MEAN_FACE_PATH
            ): (input_dir, filename)
            for input_dir, filename, output_dir in selected_videos
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing AVLips..."):
            input_dir, filename = futures[future]
            try:
                result = future.result()
                if not result:
                    print(f"[WARN] Failed to process video: {os.path.join(input_dir, filename)}")
            except Exception as e:
                print(f"[ERROR] Error in video {os.path.join(input_dir, filename)}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess videos for FakeAVCeleb, AV1M, or AVLips dataset")
    parser.add_argument('--dataset', default='AV1M', help='Select dataset: FakeAVCeleb (favc), AV1M (av1m), or AVLips')
    parser.add_argument('--split', default='train', help='For AV1M: data split to process (e.g., val, train)')
    parser.add_argument("--metadata", type=str, default="av1m_metadata/real_train_metadata.csv", help="Path to the dataset metadata")
    parser.add_argument('--category', choices=['RealVideo-RealAudio', 'RealVideo-FakeAudio', 'FakeVideo-RealAudio', 'FakeVideo-FakeAudio', 'all'], default='all', help='For FakeAVCeleb: select category (RealVideo-RealAudio, etc.)')
    parser.add_argument('--data_path', default="av1m/", help='Path to the dataset root folder')
    parser.add_argument('--max_workers', type=int, default=8, help='Number of parallel workers (default: 8)')
    parser.add_argument('--save_path', default="av1m_preprocessed/", help='Path to save avhubert preprocess outputs (lips crop and audio)')
    args = parser.parse_args()

    if args.dataset == 'FakeAVCeleb':
        if args.category == 'all':
            categories = ['RealVideo-RealAudio', 'RealVideo-FakeAudio', 'FakeVideo-RealAudio', 'FakeVideo-FakeAudio']
        elif args.category:
            categories = [args.category]

        for category in categories:
            process_fakeavceleb(category, args.metadata, args.data_path, args.save_path, args.max_workers)

    elif args.dataset == 'AV1M':
        path_to_images_root = args.data_path
        save_path = args.save_path
        process_av1m(args.metadata, path_to_images_root, save_path, args.max_workers)

    elif args.dataset == 'AVLips':
        process_avlips(args.data_path, args.save_path, args.max_workers)

if __name__ == "__main__":
    main()
