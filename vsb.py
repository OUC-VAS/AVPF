import argparse
import os
import random
import subprocess
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import dlib
from tqdm import tqdm

# ------------------------ Helpers ------------------------ #
VIDEO_EXTS = ("mp4", "mkv", "avi", "mov")

def parse_range(value: str) -> Union[float, Tuple[float, float]]:
    """Parse range argument, supporting either a single value or 'min,max' string."""
    if ',' in value:
        parts = value.split(',')
        if len(parts) != 2:
            raise ValueError(f"Invalid range format: {value}. Expected 'min,max'")
        try:
            min_val = float(parts[0])
            max_val = float(parts[1])
            if min_val > max_val:
                min_val, max_val = max_val, min_val
            return (min_val, max_val)
        except ValueError:
            raise ValueError(f"Invalid range values: {value}")
    else:
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"Invalid value: {value}")

def get_random_from_range(range_val: Union[float, Tuple[float, float]]) -> float:
    """Select a random float from a range, or return the value if it's a single float."""
    if isinstance(range_val, tuple):
        return random.uniform(range_val[0], range_val[1])
    return range_val
        
def make_sample_seed(base_seed: Optional[int], in_root: Path, src: Path) -> Optional[int]:
    """
    Generate a stable, sample-level random seed based on the global base_seed 
    and the relative path of the input video.
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

def ensure_dir(p: Path): 
    p.mkdir(parents=True, exist_ok=True)

def iter_videos(p: Path, exts: Tuple[str, ...]) -> List[Path]:
    if p.is_file(): return [p]
    vids = []
    for e in exts: vids.extend(p.rglob(f"*.{e}"))
    return vids

def read_video_bgr(path: str):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    ok, bgr = cap.read()
    while ok:
        frames.append(bgr.copy())
        ok, bgr = cap.read()
    cap.release()
    return frames, float(fps)

def write_video_bgr(frames: List[np.ndarray], out_path: Path, fps: float):
    if not frames: return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    for f in frames: vw.write(f.astype(np.uint8))
    vw.release()

def mux_audio_keep(src_video: Path, silent_mp4: Path, out_mp4: Path,
                   ffmpeg_bin="/usr/bin/ffmpeg", audio_codec="aac", overwrite=False):
    """Mux the audio track from the original video with the processed silent video."""
    cmd = [
        ffmpeg_bin, "-y" if overwrite else "-n",
        "-i", str(silent_mp4),
        "-i", str(src_video),
        "-map", "0:v:0", "-map", "1:a?",
        "-c:v", "copy", "-c:a", audio_codec,
        "-shortest",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def any_output_exists(out_dir: Path, stem: str, suffix: str, exts=VIDEO_EXTS) -> Optional[Path]:
    """Check if any output video with the given stem and suffix already exists."""
    for e in exts:
        p = out_dir / f"{stem}{suffix}.{e}"
        if p.exists():
            return p
    return None

# ------------------------ Spatial Masking ------------------------ #
class Dlib68:
    def __init__(self, predictor_path: str):
        self.det = dlib.get_frontal_face_detector()
        self.sp  = dlib.shape_predictor(predictor_path)
        
    def landmarks(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        rects = self.det(gray, 0)
        if not rects: return None
        r = max(rects, key=lambda x: x.width() * x.height())
        shape = self.sp(gray, r)
        pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], np.int32)
        return pts

def hull_mask_from_landmarks(shapeHW, pts):
    H, W = shapeHW
    mask = np.zeros((H, W), np.float32)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 1.0)
    return mask

def elastic_deform(mask, alpha=20.0, sigma=6.0):
    H, W = mask.shape
    dx = cv2.GaussianBlur((np.random.rand(H, W) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(H, W) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    deformed = cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return np.clip(deformed, 0.0, 1.0)

def sbi_mask_from_frame(bgr, dlib68: Dlib68,
                        k1_choices=(11, 15, 21), k2_choices=(21, 31, 41),
                        do_elastic=True, alpha=20.0, sigma=6.0):
    H, W = bgr.shape[:2]
    pts = dlib68.landmarks(bgr)
    if pts is None:
        # Fallback to lower face ellipse
        m = np.zeros((H, W), np.float32)
        cx, cy = int(0.5 * W), int(0.65 * H)
        rx, ry = int(0.38 * W), int(0.32 * H)
        cv2.ellipse(m, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
    else:
        m = hull_mask_from_landmarks((H, W), pts)

    if do_elastic:
        m = elastic_deform(m, alpha=alpha, sigma=sigma)

    k1 = int(random.choice(k1_choices)) | 1
    m1 = cv2.GaussianBlur(m, (k1, k1), 0)
    m1[m1 < 1.0] = 0.0

    k2 = int(random.choice(k2_choices)) | 1
    m2 = cv2.GaussianBlur(m1, (k2, k2), 0)
    m2 = np.clip(m2, 0.0, 1.0)

    r = random.choice([0.25, 0.5, 0.75, 1.0, 1.0, 1.0])
    return (m2 * r).astype(np.float32)

def gaussian_blur_mask(mask: np.ndarray, feather: int) -> np.ndarray:
    k = int(max(1, feather) // 2 * 2 + 1)
    if k > 1: mask = cv2.GaussianBlur(mask, (k, k), 0)
    mask = np.clip(mask, 0.0, 1.0)
    return mask

def make_mask(shape: Tuple[int,int], kind: str,
              feather: int = 31,
              ellipse_cx=0.5, ellipse_cy=0.65, ellipse_rx=0.35, ellipse_ry=0.28, ellipse_angle=0.0,
              dlib_predictor: Optional[str] = None) -> np.ndarray:
    """Return an HxW float32 mask in [0,1]."""
    H, W = shape
    m = np.zeros((H, W), np.float32)
    kind = kind.lower()

    if kind == "full":
        m[...] = 1.0
        return gaussian_blur_mask(m, feather)

    if kind in ("ellipse", "lowerface"):
        if kind == "lowerface":
            ellipse_cx, ellipse_cy, ellipse_rx, ellipse_ry, ellipse_angle = 0.5, 0.65, 0.38, 0.32, 0.0
        cx, cy = int(W * ellipse_cx), int(H * ellipse_cy)
        ax, ay = int(W * ellipse_rx), int(H * ellipse_ry)
        angle = ellipse_angle
        cv2.ellipse(m, (cx, cy), (ax, ay), angle, 0, 360, 1.0, -1)
        return gaussian_blur_mask(m, feather)

    if kind in ("mouth", "face"):
        return make_mask(shape, "lowerface", feather)

    return make_mask(shape, "lowerface", feather)

# --------------------- Core Blending ---------------------- #
def time_shift_frames(frames: List[np.ndarray], k: int) -> List[np.ndarray]:
    """Shift frames chronologically. k>0: earlier; k<0: later."""
    T = len(frames)
    if T == 0: return []
    idx = np.arange(T) - k
    idx = np.clip(idx, 0, T - 1)
    return [frames[i] for i in idx]

def fuse_with_mask(orig: List[np.ndarray], shifted: List[np.ndarray],
                   mask: np.ndarray, alpha: float) -> List[np.ndarray]:
    T = len(orig)
    m = (mask.astype(np.float32) * float(alpha))[:, :, None]
    inv = 1.0 - m
    out = []
    for t in range(T):
        a = shifted[t].astype(np.float32)
        b = orig[t].astype(np.float32)
        out.append(np.uint8(a * m + b * inv))
    return out

def build_temporal_gate(T: int, fps: float,
                        win_secs: float, hop_secs: float,
                        win_prob: float, smooth_frames: int) -> np.ndarray:
    """
    Generate a 1D temporal gate sequence in [0,1] to randomly mark forgery windows.
    Applies linear fade-in/fade-out smoothing if smooth_frames > 0.
    """
    g = np.zeros((T,), np.float32)
    if T == 0:
        return g

    win = max(1, int(round(win_secs * fps)))
    hop = max(1, int(round(hop_secs * fps)))

    start = 0
    while start < T:
        end = min(T, start + win)
        if random.random() < max(0.0, min(1.0, win_prob)):
            g[start:end] = 1.0
        start += hop

    s = max(0, int(smooth_frames))
    if s > 0:
        ramp = np.arange(1, s + 1, dtype=np.float32)
        kern = np.concatenate([ramp, [float(s + 1)], ramp[::-1]]).astype(np.float32)
        kern /= kern.sum()
        g = np.convolve(g, kern, mode="same").astype(np.float32)
        g = np.clip(g, 0.0, 1.0)

    return g

# ---------------------- Worker Node ------------------------- #
def process_one(src: Path, out_dir: Path, suffix: str,
                shift_cfg, mask_kind: str, feather: int, alpha: float,
                ellipse_params, ffmpeg_bin, audio_codec, overwrite: bool,
                shape_predictor: str, win_secs_range: float, hop_secs: float,
                win_prob: float, win_smooth: int, sample_seed: Optional[int]):

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
        if not frames:
            return ("SKIP", str(src), "empty video")

        arg_shift_sec, arg_shift_frames, bidir, prob = shift_cfg
        prob = max(0.0, min(1.0, float(prob)))
        use_frames = (arg_shift_frames is not None) and (str(arg_shift_frames).strip() != "")

        if random.random() > prob:
            k = 0
        else:
            if use_frames and str(arg_shift_frames).strip().lower() != "none":
                s = str(arg_shift_frames).strip()
                if s.lower().startswith("rand:"):
                    a, b = map(float, s.split(":", 1)[1].split(","))
                    lo = int(round(min(a, b)))
                    hi = int(round(max(a, b)))
                    if lo > hi:
                        lo, hi = hi, lo
                    v = random.randint(lo, hi) if hi >= lo else lo
                    if bidir and random.random() < 0.5:
                        v = -v
                    k = int(v)
                else:
                    k = int(round(float(s)))
                    if bidir and random.random() < 0.5:
                        k = -k
            else:
                s = str(arg_shift_sec).strip()
                if s.lower() == "none":
                    shift_s = 0.0
                elif s.lower().startswith("rand:"):
                    a, b = map(float, s.split(":", 1)[1].split(","))
                    v = random.uniform(min(a, b), max(a, b))
                    shift_s = (-v if (bidir and random.random() < 0.5) else v)
                else:
                    shift_s = float(s)
                k = int(round(shift_s * fps))
                
        shift_s = float(k) / float(fps) if fps else 0.0
        shifted = time_shift_frames(frames, k)

        gate = build_temporal_gate(
            T=len(frames), fps=fps,
            win_secs=get_random_from_range(win_secs_range),
            hop_secs=hop_secs,
            win_prob=win_prob,
            smooth_frames=win_smooth
        )

        dlib68 = Dlib68(shape_predictor)
        K = 3
        m_prev = None
        blended = []
        for t, (orig, shft) in enumerate(zip(frames, shifted)):
            if (t % K == 0) or (m_prev is None):
                m_prev = sbi_mask_from_frame(
                    orig, dlib68,
                    k1_choices=(11, 15, 21),
                    k2_choices=(21, 31, 41),
                    do_elastic=True, alpha=20.0, sigma=6.0
                )

            gt = float(gate[t])
            if gt <= 0.0:
                blended.append(orig.copy())
                continue

            m = (m_prev * float(alpha) * gt)[:, :, None] 
            inv = 1.0 - m
            frame_out = (shft.astype(np.float32) * m +
                         orig.astype(np.float32) * inv).astype(np.uint8)
            blended.append(frame_out)

        tmp = out_path.with_suffix(".tmp.mp4")
        write_video_bgr(blended, tmp, fps)
        mux_audio_keep(src, tmp, out_path, ffmpeg_bin=ffmpeg_bin, audio_codec=audio_codec, overwrite=overwrite)
        try: tmp.unlink()
        except: pass

        return ("OK", str(src), f"{out_path} (shift={shift_s:+.3f}s, k={k})")
    except subprocess.CalledProcessError as e:
        return ("ERR", str(src), f"ffmpeg failed: {e}")
    except Exception as e:
        return ("ERR", str(src), str(e))

# ------------------------- CLI ------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Time self-blended video pipeline (temporal shift + masked fusion) with preserved audio track.")
    ap.add_argument("--input", required=True, type=Path, help="Input directory or video path")
    ap.add_argument("--output", required=True, type=Path, help="Output directory")
    ap.add_argument("--ext", nargs="+", default=list(VIDEO_EXTS), help="Video extensions to process")
    ap.add_argument("--mirror", action="store_true", help="Mirror input directory structure to the output directory")
    ap.add_argument("--suffix", type=str, default="_tsbm", help="Output filename suffix (default: _tsbm)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")

    ap.add_argument("--shift", type=str, default="rand:0.02,0.05",
                    help="Time shift in seconds (e.g., none | <seconds> | rand:min,max)")
    ap.add_argument("--shift-frames", type=str, default="",
                    help="Time shift in frames (e.g., none | <frames> | rand:min,max). Prioritized over --shift")
    ap.add_argument("--bidir", action="store_true", help="Allow bidirectional random shift if rand is used")
    ap.add_argument("--shift-prob", type=float, default=1.0, help="Probability of applying the time shift [0,1]")

    ap.add_argument("--mask", type=str, default="lowerface",
                    choices=["full","ellipse","lowerface","mouth","face"],
                    help="Mask type (fallback if dynamic facial masking fails)")
    ap.add_argument("--feather", type=int, default=31, help="Gaussian blur kernel size for the mask (must be odd)")
    ap.add_argument("--alpha", type=float, default=1.0, help="Shifted frame blending weight (1.0 = fully shifted, 0.5 = 50/50 blend)")
    ap.add_argument("--ellipse-cx", type=float, default=0.5)
    ap.add_argument("--ellipse-cy", type=float, default=0.65)
    ap.add_argument("--ellipse-rx", type=float, default=0.38)
    ap.add_argument("--ellipse-ry", type=float, default=0.32)
    ap.add_argument("--ellipse-angle", type=float, default=0.0)

    ap.add_argument("--workers", type=int, default=4, help="Number of parallel workers (0 = auto via cpu_count)")
    ap.add_argument("--ffmpeg", type=str, default="/usr/bin/ffmpeg", help="Path to ffmpeg binary")
    ap.add_argument("--audio-codec", type=str, default="aac", help="Audio codec to use during muxing")

    ap.add_argument("--shape-predictor", type=str, required=True,
                    help="Path to shape_predictor_68_face_landmarks.dat")

    ap.add_argument("--win-secs", type=str, default="2.0",
                help="Temporal window length in seconds (fixed float or 'min,max' range)")
    ap.add_argument("--hop-secs", type=float, default=1.0,
                    help="Sliding step for the temporal window in seconds")
    ap.add_argument("--win-prob", type=float, default=0.6,
                    help="Probability of marking a temporal window for forgery [0,1]")
    ap.add_argument("--win-smooth", type=int, default=6,
                    help="Fade-in/fade-out frames at window edges to avoid hard cuts (0 = disable)")

    ap.add_argument("--seed", type=int, default=42,
                    help="Global random seed. Yields deterministic augmentation per sample via relative path hashing")

    args = ap.parse_args()

    ensure_dir(args.output)
    vids = iter_videos(args.input, tuple(args.ext))
    if not vids:
        print("No videos found to process.")
        return

    base_seed = args.seed
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

    shift_cfg = (args.shift, args.shift_frames, args.bidir, args.shift_prob)
    ellipse_params = (args.ellipse_cx, args.ellipse_cy, args.ellipse_rx, args.ellipse_ry, args.ellipse_angle)

    workers = args.workers if args.workers and args.workers > 0 else (os.cpu_count() or 1)
    workers = max(1, workers)

    try:
        win_secs_range = parse_range(args.win_secs)
    except ValueError as e:
        print(f"Error parsing --win-secs: {e}")
        return
        
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(
                    process_one,
                    v, d, args.suffix, shift_cfg,
                    args.mask, args.feather, args.alpha, ellipse_params,
                    args.ffmpeg, args.audio_codec, args.overwrite,
                    args.shape_predictor, win_secs_range, args.hop_secs,
                    args.win_prob, args.win_smooth,
                    sample_seed
                )
                for (v, d, sample_seed) in jobs
            ]

            with tqdm(total=len(futs), desc="Processing videos") as pbar:
                for fu in as_completed(futs):
                    status, vin, vout = fu.result()
                    pbar.write(f"[{status}] {vin} -> {vout}")
                    pbar.update(1)

    else:
        with tqdm(total=len(jobs), desc="Processing videos") as pbar:
            for (v, d, sample_seed) in jobs:
                status, vin, vout = process_one(
                    v, d, args.suffix, shift_cfg,
                    args.mask, args.feather, args.alpha, ellipse_params,
                    args.ffmpeg, args.audio_codec, args.overwrite,
                    args.shape_predictor, win_secs_range, args.hop_secs,
                    args.win_prob, args.win_smooth,
                    sample_seed
                )
                pbar.write(f"[{status}] {vin} -> {vout}")
                pbar.update(1)

if __name__ == "__main__":
    main()
