import argparse
import os
import random
import subprocess
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import soundfile as sf
from tqdm import tqdm

VIDEO_EXTS = ("mp4", "mkv", "avi", "mov")


# ------------------------ Utility Functions ------------------------

def ensure_dir(p: Path) -> None:
    """Ensure the specified directory exists."""
    p.mkdir(parents=True, exist_ok=True)


def iter_videos(p: Path, exts: Tuple[str, ...]) -> List[Path]:
    """Iterate over a directory or file and return a list of video paths."""
    if p.is_file():
        return [p]
    vids: List[Path] = []
    for e in exts:
        vids.extend(p.rglob(f"*.{e}"))
    return vids


def read_video_bgr(path: str) -> Tuple[List[np.ndarray], float]:
    """Read a video and return a list of BGR frames along with its FPS."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames: List[np.ndarray] = []
    ok, bgr = cap.read()
    while ok:
        frames.append(bgr.copy())
        ok, bgr = cap.read()
    cap.release()
    return frames, float(fps)


def write_video_bgr(frames: List[np.ndarray], out_path: Path, fps: float) -> None:
    """Write a list of BGR frames to a silent MP4 video."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    for f in frames:
        vw.write(f.astype(np.uint8))
    vw.release()


def any_output_exists(out_dir: Path, stem: str, suffix: str,
                      exts: Tuple[str, ...] = VIDEO_EXTS) -> Optional[Path]:
    """Check if any output video with the given stem and suffix already exists."""
    for e in exts:
        p = out_dir / f"{stem}{suffix}.{e}"
        if p.exists():
            return p
    return None


# ------------------------ Random Seed Generation ------------------------

def make_sample_seed(base_seed: Optional[int], in_root: Path, src: Path) -> Optional[int]:
    """
    Generate a stable, sample-level random seed based on the global base_seed 
    and the relative path of the input video for reproducible augmentations.
    """
    if base_seed is None:
        return None

    try:
        rel = src.resolve().relative_to(in_root)
    except Exception:
        rel = Path(src.name)

    rel_str = str(rel).replace(os.sep, "/")
    h = int.from_bytes(hashlib.sha256(rel_str.encode("utf-8")).digest()[:8], "big")

    return (h ^ int(base_seed)) & 0xFFFFFFFF


# ------------------------ Frame Similarity Validation ------------------------

def frame_mean_abs_diff(f1: np.ndarray, f2: np.ndarray, use_gray: bool = True) -> float:
    """Calculate the mean absolute difference between two frames."""
    if use_gray:
        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(g1, g2)
    else:
        diff = cv2.absdiff(f1, f2)
    return float(diff.mean())


def pick_source_idx_with_similarity(
    frames: List[np.ndarray],
    target_idx: int,
    min_gap_frames: int,
    max_gap_frames: int,
    max_mean_diff: float,
    max_trials: int = 30,
    use_gray: bool = True,
) -> Optional[int]:
    """
    Select a source frame that is visually similar to the target frame within the temporal constraints.
    Returns the index of the closest matching frame if the exact threshold is not met after max_trials.
    """
    T = len(frames)
    if T <= 1:
        return None

    candidates = [
        i for i in range(T)
        if i != target_idx and min_gap_frames <= abs(i - target_idx) <= max_gap_frames
    ]
    if not candidates:
        return None

    best_i: Optional[int] = None
    best_diff = 1e9
    for _ in range(max_trials):
        i = random.choice(candidates)
        d = frame_mean_abs_diff(frames[i], frames[target_idx], use_gray=use_gray)
        if d < best_diff:
            best_diff = d
            best_i = i
        if d <= max_mean_diff:
            return i

    return best_i


# ------------------------ Audio Processing ------------------------

def extract_audio_to_wav(
    src_video: Path,
    wav_path: Path,
    ffmpeg_bin: str = "/usr/bin/ffmpeg",
    sr: int = 16000
) -> None:
    """Extract the audio track from a video to a mono WAV file using ffmpeg."""
    cmd = [
        ffmpeg_bin, "-y",
        "-i", str(src_video),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-ac", "1",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def copy_move_audio_one_frame(
    wav_in: Path,
    wav_out: Path,
    t_src: float,
    t_tgt: float,
    frame_sec: float,
    fade_sec: float = 0.002,
) -> None:
    """
    Copy an audio segment of one frame's duration from t_src to t_tgt,
    applying a brief cross-fade to prevent audio clicks.
    """
    y, sr = sf.read(str(wav_in))
    mono = False
    if y.ndim == 1:
        y = y[:, None]
        mono = True

    n_samples = y.shape[0]
    seg_len = max(1, int(round(frame_sec * sr)))

    # Define source segment boundaries
    src_start = int(round(t_src * sr))
    src_start = max(0, min(src_start, n_samples - seg_len))
    src_end = src_start + seg_len
    seg = y[src_start:src_end].copy()

    # Define target segment boundaries
    tgt_start = int(round(t_tgt * sr))
    tgt_start = max(0, min(tgt_start, n_samples - seg_len))
    tgt_end = tgt_start + seg_len
    dst = y[tgt_start:tgt_end].copy()

    # Apply cross-fade
    fade_len = max(1, int(round(fade_sec * sr)))
    fade_len = min(fade_len, seg_len // 2)

    w = np.ones((seg_len, 1), dtype=np.float32)
    ramp_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32).reshape(-1, 1)
    w[:fade_len] = ramp_in

    mixed = (1.0 - w) * dst + w * seg
    y[tgt_start:tgt_end] = mixed

    y_out = y.squeeze() if mono else y
    sf.write(str(wav_out), y_out, sr)


def mux_video_audio(
    video_silent: Path,
    audio_wav: Path,
    out_mp4: Path,
    ffmpeg_bin: str = "/usr/bin/ffmpeg",
    audio_codec: str = "aac",
    overwrite: bool = False
) -> None:
    """Mux a silent video and a WAV audio file into the final MP4 output."""
    cmd = [
        ffmpeg_bin, "-y" if overwrite else "-n",
        "-i", str(video_silent),
        "-i", str(audio_wav),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", audio_codec,
        "-shortest",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ------------------------ Core Pipeline ------------------------

def process_one_copy_move_av(
    src: Path,
    out_dir: Path,
    suffix: str,
    min_gap_sec: float,
    max_gap_sec: float,
    max_mean_diff: float,
    gen_mask: bool,
    ffmpeg_bin: str,
    audio_codec: str,
    overwrite: bool,
    sample_seed: Optional[int],
) -> Tuple[str, str, str]:
    """
    Process a single video for AV copy-move forgery:
    1. Select a random target frame.
    2. Find a similar source frame within [min_gap_sec, max_gap_sec].
    3. Overwrite target frame with source frame (visual copy-move).
    4. Move the corresponding audio segment from source to target.
    """
    try:
        if sample_seed is not None:
            random.seed(sample_seed)
            np.random.seed(sample_seed)

        ensure_dir(out_dir)
        existed = any_output_exists(out_dir, src.stem, suffix, VIDEO_EXTS)
        if existed and not overwrite:
            return ("EXIST", str(src), str(existed))

        out_path = out_dir / f"{src.stem}{suffix}.mp4"

        frames, fps = read_video_bgr(str(src))
        T = len(frames)
        if T <= 1:
            return ("SKIP", str(src), "Video too short")

        h, w = frames[0].shape[:2]
        frame_sec = 1.0 / fps

        min_gap_frames = max(1, int(round(min_gap_sec * fps)))
        max_gap_frames = max(min_gap_frames, int(round(max_gap_sec * fps)))

        target_idx = random.randint(0, T - 1)

        source_idx = pick_source_idx_with_similarity(
            frames,
            target_idx,
            min_gap_frames=min_gap_frames,
            max_gap_frames=max_gap_frames,
            max_mean_diff=max_mean_diff,
            max_trials=30,
            use_gray=True,
        )
        
        if source_idx is None:
            return ("SKIP", str(src), "No valid source frame under temporal constraints")

        # Visual full-frame copy-move
        source_frame = frames[source_idx]
        target_frame = source_frame.copy()

        tampered_frames: List[np.ndarray] = []
        masks: Optional[List[np.ndarray]] = None
        
        if gen_mask:
            masks = [np.zeros((h, w), dtype=np.uint8) for _ in range(T)]

        for i, f in enumerate(frames):
            if i == target_idx:
                tampered_frames.append(target_frame)
                if gen_mask and masks is not None:
                    masks[i][:, :] = 255  # Entire frame is marked as tampered
            else:
                tampered_frames.append(f)

        tmp_video = out_path.with_suffix(".tmp.v.mp4")
        write_video_bgr(tampered_frames, tmp_video, fps)

        tmp_audio_raw = out_path.with_suffix(".raw.wav")
        tmp_audio_cm = out_path.with_suffix(".cm.wav")

        extract_audio_to_wav(src, tmp_audio_raw, ffmpeg_bin=ffmpeg_bin)

        # Audio one-frame copy-move
        t_src = source_idx / fps
        t_tgt = target_idx / fps
        copy_move_audio_one_frame(
            tmp_audio_raw,
            tmp_audio_cm,
            t_src=t_src,
            t_tgt=t_tgt,
            frame_sec=frame_sec,
            fade_sec=0.002,
        )

        mux_video_audio(
            tmp_video,
            tmp_audio_cm,
            out_path,
            ffmpeg_bin=ffmpeg_bin,
            audio_codec=audio_codec,
            overwrite=overwrite,
        )

        # Export masks
        if gen_mask and masks is not None:
            mask_dir = out_dir / f"{src.stem}{suffix}_masks"
            ensure_dir(mask_dir)
            for idx, m in enumerate(masks):
                cv2.imwrite(str(mask_dir / f"mask_{idx:05d}.png"), m)

        # Cleanup temporary files
        for p in [tmp_video, tmp_audio_raw, tmp_audio_cm]:
            try:
                p.unlink()
            except Exception:
                pass

        info = f"{out_path} (target={target_idx}, source={source_idx}, gap_frames={abs(source_idx-target_idx)})"
        return ("OK", str(src), info)

    except subprocess.CalledProcessError as e:
        return ("ERR", str(src), f"ffmpeg failed: {e}")
    except Exception as e:
        return ("ERR", str(src), str(e))


# ------------------------ CLI & Main ------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audio-Visual Negative Sample Generator (Full-frame visual copy-move + Aligned audio copy-move)"
    )
    ap.add_argument("--input", required=True, type=Path,
                    help="Input video file or directory")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output directory")
    ap.add_argument("--ext", nargs="+", default=list(VIDEO_EXTS),
                    help="List of video extensions to process")
    ap.add_argument("--mirror", action="store_true",
                    help="Mirror input directory structure in the output directory")
    ap.add_argument("--suffix", type=str, default="_cmav",
                    help="Output filename suffix (default: _cmav)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing output files")

    ap.add_argument("--min-gap-sec", type=float, default=0.2,
                    help="Minimum time gap between source and target frames in seconds")
    ap.add_argument("--max-gap-sec", type=float, default=0.5,
                    help="Maximum time gap between source and target frames in seconds")

    ap.add_argument("--max-mean-diff", type=float, default=10.0,
                    help="Mean absolute difference threshold for frame similarity (0~255, lower is more similar)")
    ap.add_argument("--gen-mask", action="store_true",
                    help="Export frame-by-frame tampering masks (grayscale 0/255 png)")

    ap.add_argument("--workers", type=int, default=4,
                    help="Number of parallel workers (0 or negative = number of CPU cores)")
    ap.add_argument("--ffmpeg", type=str, default="/usr/bin/ffmpeg",
                    help="Path to ffmpeg executable")
    ap.add_argument("--audio-codec", type=str, default="aac",
                    help="Output audio codec (passed to ffmpeg as -c:a)")

    ap.add_argument("--seed", type=int, default=42,
                    help="Global random seed. Derives stable sample-level seeds based on relative paths.")

    args = ap.parse_args()

    ensure_dir(args.output)
    vids = iter_videos(args.input, tuple(args.ext))
    if not vids:
        print("No videos found.")
        return

    base_seed: Optional[int] = args.seed
    in_root = args.input.resolve()
    in_is_dir = in_root.is_dir()

    jobs = []
    for v in vids:
        out_dir = args.output
        if args.mirror and in_is_dir:
            try:
                rel = v.resolve().parent.relative_to(in_root)
                out_dir = args.output / rel
            except Exception:
                pass

        existed = any_output_exists(out_dir, v.stem, args.suffix, VIDEO_EXTS)
        if existed and not args.overwrite:
            print(f"[EXIST] {v} -> {existed}")
            continue

        sample_seed = make_sample_seed(base_seed, in_root, v)
        jobs.append((v, out_dir, sample_seed))

    if not jobs:
        print("All outputs already exist; process complete.")
        return

    workers = args.workers if args.workers and args.workers > 0 else (os.cpu_count() or 1)
    workers = max(1, workers)

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(
                    process_one_copy_move_av,
                    v, d, args.suffix,
                    args.min_gap_sec, args.max_gap_sec,
                    args.max_mean_diff,
                    args.gen_mask,
                    args.ffmpeg, args.audio_codec, args.overwrite,
                    sample_seed,
                )
                for (v, d, sample_seed) in jobs
            ]
            with tqdm(total=len(futs), desc="Copy-move AV videos") as pbar:
                for fu in as_completed(futs):
                    status, vin, vout = fu.result()
                    pbar.write(f"[{status}] {vin} -> {vout}")
                    pbar.update(1)
    else:
        with tqdm(total=len(jobs), desc="Copy-move AV videos") as pbar:
            for (v, d, sample_seed) in jobs:
                status, vin, vout = process_one_copy_move_av(
                    v, d, args.suffix,
                    args.min_gap_sec, args.max_gap_sec,
                    args.max_mean_diff,
                    args.gen_mask,
                    args.ffmpeg, args.audio_codec, args.overwrite,
                    sample_seed,
                )
                pbar.write(f"[{status}] {vin} -> {vout}")
                pbar.update(1)


if __name__ == "__main__":
    main()
