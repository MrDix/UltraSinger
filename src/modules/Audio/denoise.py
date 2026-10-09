"""Reduce noise from audio"""

import os
import tempfile

import ffmpeg
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import correlate

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, gold_highlighted, green_highlighted
from modules.os_helper import check_file_exists

# The afftdn filter delays its output (by 25 ms in ffmpeg 5 to 8, at any sample
# rate): it pads the start and drops the end. Left in, everything analysed on the
# denoised vocal (transcription, pitch, silence) would lag the song by that much.
MAX_FILTER_DELAY_S = 0.2
_EXCERPT_S = 30.0  # the delay is measured on the loudest stretch of this length
_MIN_CORRELATION = 0.5  # below this the match is not trusted and nothing is shifted


def __ffmpeg_reduce_noise(input_file_path: str, output_file: str,
                          noise_reduction: float = 20,
                          noise_floor: float = -80,
                          track_noise: bool = True) -> None:
    """Reduce noise from vocal audio with ffmpeg.

    Uses the afftdn (FFT-based denoising) filter.

    Args:
        input_file_path: Path to input audio file.
        output_file: Path for denoised output file.
        noise_reduction: Noise reduction in dB (0.01-97). Default: 20.
            Lower values preserve more vocal detail (consonants, sibilants).
            Higher values remove more noise but risk destroying vocal nuances.
        noise_floor: Noise floor in dB (-80 to -20). Default: -80.
        track_noise: Enable adaptive noise floor tracking. Default: True.
    """

    tn_flag = 1 if track_noise else 0
    af_filter = f"afftdn=nr={noise_reduction}:nf={noise_floor}:tn={tn_flag}"

    print(
        f"{ULTRASINGER_HEAD} Reduce noise from vocal audio with {blue_highlighted('ffmpeg')} (nr={noise_reduction}dB, nf={noise_floor}dB)."
    )
    try:
        (
            ffmpeg.input(input_file_path)
            .output(output_file, af=af_filter)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as ffmpeg_exception:
        print("ffmpeg stdout:", ffmpeg_exception.stdout.decode("utf8"))
        print("ffmpeg stderr:", ffmpeg_exception.stderr.decode("utf8"))
        raise ffmpeg_exception


def filter_delay(original: np.ndarray, filtered: np.ndarray, sample_rate: int,
                 max_delay_s: float = MAX_FILTER_DELAY_S) -> int:
    """Samples by which the mono signal ``filtered`` lags ``original``.

    Cross-correlates the loudest stretch of the original with the filtered
    signal for delays from 0 to ``max_delay_s``. Returns 0 when the signals
    are too short or do not match clearly.
    """
    max_lag = int(max_delay_s * sample_rate)
    usable = min(len(original), len(filtered)) - max_lag
    if usable <= 0:
        return 0
    length = min(usable, int(_EXCERPT_S * sample_rate))
    energy = np.concatenate([[0.0], np.cumsum(original[:usable].astype(np.float64) ** 2)])
    starts = np.arange(0, usable - length + 1, sample_rate)
    a = int(starts[np.argmax(energy[starts + length] - energy[starts])])
    x = original[a:a + length].astype(np.float64)
    y = filtered[a:a + length + max_lag].astype(np.float64)
    c = correlate(y, x, mode="valid", method="fft")  # c[k] = sum(x[i] * y[i + k])
    k = int(np.argmax(c))
    norm = np.sqrt(np.dot(x, x) * np.dot(y[k:k + length], y[k:k + length]))
    if norm <= 0 or c[k] / norm < _MIN_CORRELATION:
        return 0
    return k


def remove_filter_delay(original_path: str, filtered_path: str) -> float:
    """Move the filtered file back in line with the original; returns the delay removed (s)."""
    try:
        original, sample_rate = librosa.load(original_path, sr=None, mono=True)
        info = sf.info(filtered_path)
        filtered, filtered_rate = sf.read(filtered_path, always_2d=True, dtype="float32")
    except Exception as e:  # noqa: BLE001 - any decoder error; the denoised file is still usable
        print(f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} could not check the delay of the "
              f"noise filter ({e!r}) - the denoised vocal may lag the song")
        return 0.0
    if filtered_rate != sample_rate:
        return 0.0
    delay = filter_delay(original, filtered.mean(axis=1), sample_rate)
    if delay > 0:
        aligned = np.zeros_like(filtered)
        aligned[:len(filtered) - delay] = filtered[delay:]
        # Write next to the file and swap it in, so a failed write never leaves a broken cache file
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".wav", dir=os.path.dirname(os.path.abspath(filtered_path)))
            os.close(fd)
            sf.write(tmp_path, aligned, sample_rate, format=info.format, subtype=info.subtype)
            os.replace(tmp_path, filtered_path)
        except Exception as e:  # noqa: BLE001 - keep the (delayed) denoised file usable
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            print(f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} could not remove the delay of the "
                  f"noise filter ({e!r}) - the denoised vocal may lag the song")
            return 0.0
    return delay / sample_rate


def denoise_vocal_audio(input_path: str, output_path: str,
                        skip_cache: bool = False,
                        noise_reduction: float = 20,
                        noise_floor: float = -80,
                        track_noise: bool = True) -> None:
    """Denoise vocal audio, sample-aligned with the input"""
    cache_available = check_file_exists(output_path)
    if skip_cache or not cache_available:
        __ffmpeg_reduce_noise(input_path, output_path,
                              noise_reduction=noise_reduction,
                              noise_floor=noise_floor,
                              track_noise=track_noise)
        delay = remove_filter_delay(input_path, output_path)
        if delay:
            print(f"{ULTRASINGER_HEAD} Removed the {delay * 1000:.0f} ms delay of the noise filter")
    else:
        print(f"{ULTRASINGER_HEAD} {green_highlighted('cache')} reusing cached denoised audio")
