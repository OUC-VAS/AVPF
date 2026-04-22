import argparse
import os
import random
import subprocess
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

VIDEO_EXTS = ("mp4", "mkv", "avi", "mov")

# ------------------------ Helpers ------------------------ #
def ensure_dir(p: Path): 
    p.mkdir(parents=True, exist_ok=True)

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

def iter_videos(p: Path, exts: Tuple[str, ...]) -> List[Path]:
    if p.is_file(): return [p]
    vids = []
    for e in exts: vids.extend(p.rglob(f"*.{e}"))
    return vids

def any_output_exists(out_dir: Path, stem: str, suffix: str, exts=VIDEO_EXTS) -> Optional[Path]:
    """Check if any output video with the given stem and suffix already exists."""
    for e in exts:
        p = out_dir / f"{stem}{suffix}.{e}"
        if p.exists():
            return p
    return None

# ------------------------ FFmpeg I/O ------------------------ #
def ffmpeg_extract_audio(src_video: Path, out_wav: Path, ffmpeg_bin="/usr/bin/ffmpeg",
                         sr: int = 22050, mono: bool = True, overwrite: bool = True):
    """Extract audio from video to a WAV file."""
    cmd = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error",
        "-y" if overwrite else "-n",
        "-i", str(src_video),
        "-vn",
    ]
    if mono: cmd += ["-ac", "1"]
    if sr:   cmd += ["-ar", str(sr)]
    cmd += ["-f", "wav", str(out_wav)]
    subprocess.run(cmd, check=True)

def ffmpeg_mux_replace_audio(src_video: Path, new_audio_wav: Path, out_video: Path,
                             ffmpeg_bin="/usr/bin/ffmpeg", audio_codec="aac", overwrite: bool = False):
    """Replace the audio track of the source video with a new WAV file without re-encoding the video."""
    cmd = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error",
        "-y" if overwrite else "-n",
        "-i", str(src_video),
        "-i", str(new_audio_wav),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", audio_codec,
        "-shortest",
        str(out_video)
    ]
    subprocess.run(cmd, check=True)

# ------------------------ Mel I/O ------------------------ #
def audio_to_mel_spectrogram(
    wav_path: str,
    sr: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    power: float = 2.0,
):
    """Convert audio to Mel spectrogram."""
    y, sr = librosa.load(wav_path, sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power
    ).astype(np.float32)
    return mel, sr

def mel_to_audio(
    mel: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    power: float = 2.0,
    n_iter: int = 32,
):
    """Reconstruct audio waveform from Mel spectrogram using Griffin-Lim."""
    y = librosa.feature.inverse.mel_to_audio(
        M=mel, sr=sr, n_fft=n_fft, hop_length=hop_length, power=power, n_iter=n_iter
    )
    return y.astype(np.float32)

# ------------------------ Core Processing (ASBA) ------------------------ #
def time_shift_spectrogram(mel: np.ndarray, shift_frames: int) -> np.ndarray:
    """Shift the spectrogram along the time axis with edge replication."""
    F, T = mel.shape
    idx = np.arange(T) - int(shift_frames)
    idx = np.clip(idx, 0, T-1)
    return mel[:, idx]

def build_temporal_gate(
    T: int,
    fps_t: float,
    win_secs: float,
    hop_secs: float,
    win_prob: float,
    smooth_frames: int
) -> np.ndarray:
    """Generate a 1D temporal gate sequence in [0,1] for random forgery window masking."""
    g = np.zeros((T,), np.float32)
    if T == 0: return g

    win = max(1, int(round(win_secs * fps_t)))
    hop = max(1, int(round(hop_secs * fps_t)))

    start = 0
    while start < T:
        end = min(T, start + win)
        if random.random() < max(0.0, min(1.0, win_prob)):
            g[start:end] = 1.0
        start += hop

    s = max(0, int(smooth_frames))
    if s > 0:
        ramp = np.arange(1, s+1, dtype=np.float32)
        kern = np.concatenate([ramp, [float(s+1)], ramp[::-1]]).astype(np.float32)
        kern /= kern.sum()
        g = np.convolve(g, kern, mode="same").astype(np.float32)
        g = np.clip(g, 0.0, 1.0)

    return g

def fuse_with_gate(orig: np.ndarray, shifted: np.ndarray, gate_1d: np.ndarray, alpha: float) -> np.ndarray:
    """Fuse original and shifted spectrograms using the temporal gate."""
    ga = (np.clip(gate_1d, 0.0, 1.0).astype(np.float32) * float(alpha))[None, :] 
    return shifted * ga + orig * (1.0 - ga)

def parse_shift(arg_shift: str, bidir: bool, shift_prob: float) -> float:
    """Parse the time shift argument (seconds)."""
    if random.random() > max(0.0, min(1.0, shift_prob)):
        return 0.0

    if isinstance(arg_shift, str) and arg_shift.lower().startswith("rand:"):
        try:
            a, b = map(float, arg_shift.split(":", 1)[1].split(","))
        except Exception:
            a, b = 0.2, 0.5
        v = random.uniform(min(a, b), max(a, b))
        if bidir and (random.random() < 0.5):
            v = -v
        return float(v)

    if arg_shift.strip().lower() == "none":
        return 0.0

    try:
        return float(arg_shift)
    except Exception:
        return 0.0

# ------------------------ Worker Node ------------------------ #
def process_one(
    src_video: Path, out_dir: Path, suffix: str,
    arg_shift: str, bidir: bool, shift_prob: float,
    gate_mode: str, win_secs: float, hop_secs: float, win_prob: float, win_smooth: int,
    alpha: float,
    target_sr: int, n_mels: int, n_fft: int, hop_length: int, power: float, n_iter: int,
    ffmpeg_bin: str, audio_codec: str,
    overwrite: bool, keep_tmp: bool, mono: bool, 
    sample_seed: Optional[int],
):
    try:
        if sample_seed is not None:
            random.seed(sample_seed)
            np.random.seed(sample_seed)
            
        ensure_dir(out_dir)
        existed = any_output_exists(out_dir, src_video.stem, suffix, exts=VIDEO_EXTS)
        out_path = out_dir / f"{src_video.stem}{suffix}.mp4"
        if existed and not overwrite:
            return ("EXIST", str(src_video), str(existed))

        tmp_wav = out_dir / f".{src_video.stem}{suffix}.tmp.wav"
        ffmpeg_extract_audio(src_video, tmp_wav, ffmpeg_bin=ffmpeg_bin, sr=target_sr, mono=mono, overwrite=True)

        mel, sr = audio_to_mel_spectrogram(
            str(tmp_wav), sr=target_sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, power=power
        )
        F, T = mel.shape
        fps_t = float(sr) / float(hop_length)

        shift_s = parse_shift(arg_shift, bidir, shift_prob)
        k_frames = int(round(shift_s * fps_t))
        shifted = time_shift_spectrogram(mel, k_frames)

        if gate_mode == "sliding":
            gate = build_temporal_gate(
                T=T, fps_t=fps_t, win_secs=win_secs, hop_secs=hop_secs,
                win_prob=win_prob, smooth_frames=win_smooth
            )
        elif gate_mode == "full":
            gate = np.ones((T,), np.float32)
        else:  
            gate = np.zeros((T,), np.float32)

        blended = fuse_with_gate(mel, shifted, gate, alpha)
        y = mel_to_audio(blended, sr=sr, n_fft=n_fft, hop_length=hop_length, power=power, n_iter=n_iter)

        tmp_proc_wav = out_dir / f".{src_video.stem}{suffix}.proc.wav"
        sf.write(tmp_proc_wav, y, sr)
        ffmpeg_mux_replace_audio(src_video, tmp_proc_wav, out_path, ffmpeg_bin=ffmpeg_bin, audio_codec=audio_codec, overwrite=overwrite)

        if not keep_tmp:
            try: tmp_wav.unlink()
            except: pass
            try: tmp_proc_wav.unlink()
            except: pass

        gate_info = f"gate={gate_mode}"
        if gate_mode == "sliding":
            gate_info += (
                f"(win={win_secs:.3f}s, hop={hop_secs:.3f}s, "
                f"prob={win_prob:.2f}, smooth={win_smooth})"
            )

        detail = (
            f"{out_path} (shift={shift_s:+.3f}s, k_frames={k_frames}, "
            f"{gate_info}, alpha={alpha:.2f})"
        )

        return ("OK", str(src_video), detail)

    except subprocess.CalledProcessError as e:
        return ("ERR", str(src_video), f"ffmpeg failed: {e}")
    except Exception as e:
        return ("ERR", str(src_video), str(e))

# ------------------------ CLI ------------------------ #
def build_parser():
    ap = argparse.ArgumentParser(description="Audio Self-Blended augmentation (replaces video audio track, keeps video intact)")
    
    ap.add_argument("--input", required=True, type=Path, help="Input video or directory")
    ap.add_argument("--output", required=True, type=Path, help="Output directory")
    ap.add_argument("--ext", nargs="+", default=list(VIDEO_EXTS), help="Video extensions to match when input is a directory")
    ap.add_argument("--mirror", action="store_true", help="Mirror input directory structure to the output directory")
    ap.add_argument("--suffix", type=str, default="_asbm", help="Output filename suffix (default: _asbm)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")

    ap.add_argument("--shift", type=str, default="rand:0.2,0.5", help="Time shift in seconds (e.g., none | <seconds> | rand:min,max)")
    ap.add_argument("--bidir", action="store_true", help="Allow bidirectional random shift if rand is used")
    ap.add_argument("--shift-prob", type=float, default=1.0, help="Probability of applying the time shift [0,1]")

    ap.add_argument("--gate-mode", type=str, default="sliding", choices=["sliding", "full", "none"],
                    help="Temporal gating mode: sliding | full | none")
    ap.add_argument("--win-secs", type=float, default=2.0, help="Window length in seconds (for sliding mode)")
    ap.add_argument("--hop-secs", type=float, default=1.0, help="Sliding step in seconds (for sliding mode)")
    ap.add_argument("--win-prob", type=float, default=0.6, help="Probability of marking a window for blending [0,1] (for sliding mode)")
    ap.add_argument("--win-smooth", type=int, default=6, help="Fade-in/fade-out frames at window edges (0 = disable, for sliding mode)")

    ap.add_argument("--alpha", type=float, default=1.0, help="Blending weight: 1.0 = fully shifted, 0.5 = 50/50 blend, 0 = disabled")

    ap.add_argument("--sr", type=int, default=16000, help="Target sample rate (used for both extraction and reconstruction)")
    ap.add_argument("--n-mels", type=int, default=128, help="Number of Mel filterbanks")
    ap.add_argument("--n-fft", type=int, default=2048, help="FFT window size")
    ap.add_argument("--hop-length", type=int, default=512, help="STFT hop length")
    ap.add_argument("--power", type=float, default=2.0, help="Power of the spectrogram")
    ap.add_argument("--n-iter", type=int, default=32, help="Griffin-Lim iterations")

    ap.add_argument("--ffmpeg", type=str, default="/usr/bin/ffmpeg", help="Path to ffmpeg binary")
    ap.add_argument("--audio-codec", type=str, default="aac", help="Audio codec for muxing (e.g., aac, copy)")

    ap.add_argument("--workers", type=int, default=12, help="Number of parallel workers (0 = auto via cpu_count)")
    ap.add_argument("--keep-tmp", action="store_true", help="Keep temporary extracted wav files for debugging")
    ap.add_argument("--mono", action="store_true", help="Force mono audio extraction")
    ap.add_argument("--seed", type=int, default=42, help="Global random seed for deterministic generation")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    
    return ap

def main():
    ap = build_parser()
    args = ap.parse_args()

    ensure_dir(args.output)
    in_root = args.input.resolve()
    in_is_dir = in_root.is_dir()

    vids = iter_videos(in_root, tuple(args.ext))
    if not vids:
        print("No videos found to process.")
        return

    base_seed = args.seed 

    jobs = []
    for v in vids:
        out_dir = args.output
        if args.mirror and in_is_dir:
            try:
                rel = v.resolve().parent.relative_to(in_root)
                out_dir = args.output / rel
            except Exception:
                pass

        existed = any_output_exists(out_dir, v.stem, args.suffix, exts=VIDEO_EXTS)
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

    stats = {"OK": 0, "EXIST": 0, "ERR": 0}

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(
                process_one,
                v, d, args.suffix,
                args.shift, args.bidir, args.shift_prob,
                args.gate_mode, args.win_secs, args.hop_secs, args.win_prob, args.win_smooth,
                args.alpha,
                args.sr, args.n_mels, args.n_fft, args.hop_length, args.power, args.n_iter,
                args.ffmpeg, args.audio_codec,
                args.overwrite, args.keep_tmp, True if args.mono else True, sample_seed
            ) for (v, d, sample_seed) in jobs] 

            base_iter = as_completed(futs)

            pbar = None
            if not args.no_progress:
                pbar = tqdm(total=len(futs), desc="Processing videos",
                            unit="video", dynamic_ncols=True)

            for fu in base_iter:
                status, vin, vout = fu.result()
                stats[status] = stats.get(status, 0) + 1

                msg = f"[{status}] {vin} -> {vout}"
                if pbar is not None:
                    pbar.write(msg)
                    pbar.update(1)
                else:
                    print(msg)

    else:
        pbar = None
        if not args.no_progress:
            pbar = tqdm(jobs, total=len(jobs), desc="Processing videos",
                        unit="video", dynamic_ncols=True)
            iterable = pbar
        else:
            iterable = jobs

        for v, d, sample_seed in iterable: 
            status, vin, vout = process_one(
                v, d, args.suffix,
                args.shift, args.bidir, args.shift_prob,
                args.gate_mode, args.win_secs, args.hop_secs, args.win_prob, args.win_smooth,
                args.alpha,
                args.sr, args.n_mels, args.n_fft, args.hop_length, args.power, args.n_iter,
                args.ffmpeg, args.audio_codec,
                args.overwrite, args.keep_tmp, True if args.mono else True, sample_seed
            )
            stats[status] = stats.get(status, 0) + 1

            msg = f"[{status}] {vin} -> {vout}"
            if pbar is not None:
                pbar.write(msg)
            else:
                print(msg)

    total = sum(stats.values())
    print(f"\nDone. Total: {total} | OK: {stats.get('OK',0)} | EXIST: {stats.get('EXIST',0)} | ERR: {stats.get('ERR',0)}")

if __name__ == "__main__":
    main()
