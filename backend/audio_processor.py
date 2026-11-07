"""
Audio DSP, Vosk Forced Word Alignment & Praat TD-PSOLA Pipeline for OpenUTAU Voicebank Generation:
- Uses Vosk grammar-guided speech recognition to locate the exact millisecond start/end
  timestamps of every word the user spoke in their recording.
- Slices the user's ACTUAL spoken words and syllables ("Daisy", "Bell", "give", "me", "half",
  "crazy", "all", "for", "the", "love", "of", "you", etc.) directly from their recording—
  preserving 100% of their real regional/British accent, real consonants, and real vocal tract!
- Uses Phase-Locked Time-Domain WSOLA (`wsola_stretch_phase_locked`) + Praat TD-PSOLA
  (`praat-parselmouth` overlap-add) instead of spectral vocoding (`pyworld`), keeping the user's
  actual time-domain microphone waveform, glottal pulse phase, and natural breath texture intact.
- Studio soft-knee limiter to prevent any clipping on high notes.
"""

from __future__ import annotations

import io
import json
import math
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.interpolate as interp
import scipy.signal as signal
import soundfile as sf

from .phonemes import ALIGNMENT_VOCABULARY

try:
    import parselmouth
    from parselmouth.praat import call as praat_call
    HAS_PARSELMOUTH = True
except Exception:
    parselmouth = None
    praat_call = None
    HAS_PARSELMOUTH = False

try:
    if "pkg_resources" not in sys.modules:
        try:
            import pkg_resources  # type: ignore
        except ImportError:
            _shim = types.ModuleType("pkg_resources")
            _shim.get_distribution = lambda _name: types.SimpleNamespace(version="0.3.5")  # type: ignore
            sys.modules["pkg_resources"] = _shim
    import pyworld as pw
    HAS_PYWORLD = True
except Exception:
    pw = None
    HAS_PYWORLD = False

try:
    from vosk import KaldiRecognizer, Model, SetLogLevel
    SetLogLevel(-1)
    HAS_VOSK = True
except Exception:
    KaldiRecognizer = None
    Model = None
    HAS_VOSK = False

_VOSK_MODEL: Optional[Any] = None

SAMPLE_RATE = 44100
TARGET_NOTE_HZ = 261.63

_NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def get_vosk_model() -> Optional[Any]:
    global _VOSK_MODEL
    if not HAS_VOSK or Model is None:
        return None
    if _VOSK_MODEL is None:
        try:
            _VOSK_MODEL = Model(lang="en-us")
        except Exception:
            return None
    return _VOSK_MODEL


def midi_to_note_name(midi_num: int) -> str:
    octave = (midi_num // 12) - 1
    name = _NOTE_NAMES[midi_num % 12]
    return f"{name}{octave}"


def choose_user_base_note(median_f0_hz: float) -> Tuple[float, str, int]:
    """
    Snap the UTAU base reference pitch to the musical semitone closest to the user's
    actual speaking pitch, and compute `semitone_shift` so the center of the Daisy Bell
    melody (MIDI 66) sits right in the user's comfortable natural chest register (+1 semitone
    above their median speaking pitch).
    """
    clamped_f0 = float(np.clip(median_f0_hz, 75.0, 310.0))
    user_midi = int(round(69.0 + 12.0 * np.log2(clamped_f0 / 440.0)))
    user_midi = int(np.clip(user_midi, 40, 62))

    base_hz = float(440.0 * (2.0 ** ((user_midi - 69.0) / 12.0)))
    note_name = midi_to_note_name(user_midi)

    # Center of Daisy Bell melody is MIDI 66 (F#4).
    # Place the melody's center at `user_midi + 1` so the song stays inside [0.78x .. 1.55x] of speaking F0!
    semitone_shift = int((user_midi + 1) - 66)
    return round(base_hz, 2), note_name, semitone_shift


def declip_vocal_waveform(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Studio Cubic-Spline De-Clipper:
    - Reconstructs flat-topped microphone clipping plateaus (caused by hot OS/browser mic gain)
      using natural CubicSpline arches anchored to the unclipped slopes on either side.
    - Filters out sub-sonic rumble (< 55 Hz) and ultrasonic harshness (> 9000 Hz).
    """
    if len(audio) < 64:
        return audio.copy()
    x = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - np.mean(x)
    peak = float(np.max(np.abs(x)))
    if peak < 1e-5:
        return x.astype(np.float64)

    thresh = peak * 0.85
    is_clipped = np.abs(x) >= thresh
    clipped_mask = is_clipped.copy()
    clipped_mask[1:] |= is_clipped[:-1]
    clipped_mask[:-1] |= is_clipped[1:]

    y = x.copy()
    n = len(y)
    idx = 0
    while idx < n:
        if not clipped_mask[idx]:
            idx += 1
            continue
        start = idx
        while idx < n and clipped_mask[idx]:
            idx += 1
        end = idx
        span_len = end - start
        if 1 <= span_len <= 45 and start >= 3 and end <= n - 3:
            anchor_idx = np.array([start - 3, start - 2, start - 1, end, end + 1, end + 2], dtype=np.float64)
            anchor_val = y[anchor_idx.astype(int)]
            cs = interp.CubicSpline(anchor_idx, anchor_val, bc_type="natural")
            interp_vals = cs(np.arange(start, end, dtype=np.float64))
            y[start:end] = np.clip(interp_vals, -peak * 1.22, peak * 1.22)

    if len(y) > 128:
        b_hp, a_hp = signal.butter(2, 55.0 / (sr / 2.0), btype="high")
        b_lp, a_lp = signal.butter(2, 9200.0 / (sr / 2.0), btype="low")
        y = signal.filtfilt(b_hp, a_hp, y)
        y = signal.filtfilt(b_lp, a_lp, y)
    return y.astype(np.float64)


def soft_limit_audio(
    audio: np.ndarray,
    target_peak: float = 0.58,
    target_rms: float = 0.135,
) -> np.ndarray:
    """
    Pure Linear Studio RMS Gain Staging (ZERO `tanh` wave-shaping distortion!):
    - Sets active vocal RMS to a comfortable `-17 dBFS` (`0.135`) and linearly bounds
      peaks to `<= target_peak` (`0.58`, leaving `+4.7 dB` of clean digital headroom).
    """
    if len(audio) == 0:
        return audio
    x = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - np.mean(x)
    peak = float(np.max(np.abs(x)))
    if peak < 1e-6:
        return x.astype(np.float64)

    active = x[np.abs(x) > peak * 0.08]
    rms = float(np.sqrt(np.mean(active ** 2))) if len(active) > 16 else float(np.sqrt(np.mean(x ** 2)))
    eff_target_rms = min(target_rms, target_peak * 0.26)
    gain = eff_target_rms / max(1e-6, rms)
    scaled = x * gain

    new_peak = float(np.max(np.abs(scaled)))
    if new_peak > target_peak:
        scaled = scaled * (target_peak / new_peak)
    return scaled.astype(np.float64)


def normalize_peak(audio: np.ndarray, target_peak: float = 0.58) -> np.ndarray:
    return soft_limit_audio(audio, target_peak=target_peak)


def wsola_stretch_phase_locked(
    audio: np.ndarray,
    target_samples: int,
    sr: int = SAMPLE_RATE,
    min_head_samples: int = 1600,
) -> np.ndarray:
    """
    Phase-Locked Multi-Period Time-Domain WSOLA (Waveform-Similarity Overlap-Add):
    - Expands (or trims) the interior vowel nucleus of a spoken word in the TIME DOMAIN
      without any FFT spectral vocoding.
    - Uses 36ms periodic Hann windows (~4 glottal periods per grain) and cross-correlation
      (`np.correlate` over +-11ms) to lock every grain overlap to the exact glottal cycle phase.
    - Preserves 100% of the speaker's raw microphone waveform, glottal closure texture, breathiness,
      and regional accent formants.
    """
    n = len(audio)
    if n < int(0.075 * sr):
        return audio.copy()

    # If target_samples is shorter than the raw clip, trim the interior vowel with a phase-locked splice
    H = int(0.018 * sr)  # 18ms half-window
    W = 2 * H            # 36ms full grain
    search_tol = int(0.011 * sr)  # +-11ms covers a full glottal period down to 45Hz

    if n >= target_samples:
        if n - target_samples < H:
            return audio[:target_samples].copy()
        head_keep = max(min_head_samples, int(target_samples * 0.55))
        head_keep = min(head_keep, target_samples - H - 32)
        tail_keep = target_samples - head_keep
        if head_keep > H and tail_keep > H and n > head_keep + tail_keep + search_tol:
            idx = np.arange(W, dtype=np.float64)
            hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * idx / W))
            fade_in = hann[:H]
            fade_out = hann[H:]
            ref_seg = audio[head_keep - H : head_keep]
            t_start_nom = n - tail_keep - H
            s_min = max(head_keep, t_start_nom - search_tol)
            s_max = min(n - tail_keep - H, t_start_nom + search_tol)
            if s_max > s_min and np.max(np.abs(ref_seg)) > 1e-6:
                corr = np.correlate(audio[s_min : s_max + H], ref_seg, mode="valid")
                best_t = s_min + int(np.argmax(corr))
            else:
                best_t = t_start_nom
            overlap = audio[head_keep - H : head_keep] * fade_out + audio[best_t : best_t + H] * fade_in
            return np.concatenate([audio[: head_keep - H], overlap, audio[best_t + H : best_t + H + tail_keep]])
        return audio[:target_samples].copy()

    head_end = min(n // 2, max(min_head_samples, int(n * 0.24)))
    tail_len = min(n // 4, max(H + search_tol + 64, int(n * 0.20)))
    tail_start = n - tail_len

    if tail_start - head_end < W + search_tol:
        # Fallback for very short clips: use smaller 20ms grain
        H = int(0.010 * sr)
        W = 2 * H
        search_tol = int(0.008 * sr)
        head_end = max(int(0.018 * sr), int(n * 0.22))
        tail_len = max(H + search_tol + 16, int(n * 0.20))
        tail_start = n - tail_len
        if tail_start - head_end < W:
            return audio.copy()

    idx = np.arange(W, dtype=np.float64)
    hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * idx / W))
    fade_in = hann[:H]
    fade_out = hann[H:]

    out: List[float] = list(audio[:head_end])
    prev_tail = audio[head_end : head_end + H] * fade_out
    prev_unwindowed = audio[head_end : head_end + H]

    needed_mid = max(H, target_samples - head_end - tail_len)
    n_hops = max(1, int(round(needed_mid / H)))

    nuc_min = head_end
    nuc_max = max(nuc_min + 1, tail_start - W)

    for step in range(n_hops):
        prog = (step + 0.5) / max(1, n_hops)
        # Smooth organic traversal across the interior vowel nucleus
        # so adjacent glottal cycles are always distinct and never freeze on a single pulse
        sweep = 0.5 - 0.5 * np.cos(np.pi * prog)
        target_pos = int(nuc_min + sweep * (nuc_max - nuc_min))
        s_min = max(nuc_min, target_pos - search_tol)
        s_max = min(n - W, target_pos + search_tol)

        if s_max > s_min and np.max(np.abs(prev_unwindowed)) > 1e-6:
            cand = audio[s_min : s_max + H]
            corr = np.correlate(cand, prev_unwindowed, mode="valid")
            chosen = s_min + int(np.argmax(corr))
        else:
            chosen = target_pos

        grain = audio[chosen : chosen + W]
        overlap_sum = prev_tail + grain[:H] * fade_in
        out.extend(overlap_sum)
        prev_tail = grain[H:] * fade_out
        prev_unwindowed = grain[H:]

    # Phase-lock the natural word release tail with the final half-grain
    t_min = max(head_end, tail_start - search_tol)
    t_max = min(n - H - 1, tail_start + search_tol)
    if t_max > t_min and np.max(np.abs(prev_unwindowed)) > 1e-6:
        cand_t = audio[t_min : t_max + H]
        corr_t = np.correlate(cand_t, prev_unwindowed, mode="valid")
        best_tail = t_min + int(np.argmax(corr_t))
    else:
        best_tail = min(n - H - 1, tail_start)

    final_overlap = prev_tail + audio[best_tail : best_tail + H] * fade_in
    out.extend(final_overlap)
    out.extend(audio[best_tail + H :])
    return np.array(out, dtype=np.float64)


def psola_pitch_transfer(
    audio: np.ndarray,
    sr: int,
    pitch_curve_fn: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
    f0_floor: float = 55.0,
    f0_ceil: float = 480.0,
) -> np.ndarray:
    """
    Time-Domain Pitch-Synchronous Overlap-Add (TD-PSOLA) via Praat (`praat-parselmouth`):
    - Moves glottal waveform cycles in the time domain to match `pitch_curve_fn(times, orig_f0, med_f0)`
      without running any FFT spectral vocoder (`pyworld`).
    - Unvoiced consonants and vocal tract formants (regional accent) remain 100% untouched.
    """
    if not HAS_PARSELMOUTH or parselmouth is None or praat_call is None or len(audio) < int(0.065 * sr):
        return audio.copy()

    try:
        snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=float(sr))
        pitch_obj = snd.to_pitch(time_step=0.010, pitch_floor=f0_floor, pitch_ceiling=f0_ceil)
        orig_f0 = pitch_obj.selected_array["frequency"]
        times = pitch_obj.xs()

        voiced = orig_f0[orig_f0 > f0_floor]
        if len(voiced) == 0:
            return audio.copy()
        med_f0 = float(np.median(voiced))

        target_f0 = pitch_curve_fn(times, orig_f0, med_f0)

        manip = praat_call(snd, "To Manipulation", 0.005, f0_floor, f0_ceil)
        pitch_tier = praat_call(manip, "Extract pitch tier")
        praat_call(pitch_tier, "Remove points between", 0.0, snd.xmax)

        for t_val, f_val in zip(times, target_f0):
            if f_val > f0_floor:
                praat_call(pitch_tier, "Add point", float(t_val), float(f_val))

        praat_call([pitch_tier, manip], "Replace pitch tier")
        res_snd = praat_call(manip, "Get resynthesis (overlap-add)")
        out = res_snd.values[0].astype(np.float64)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        return audio.copy()


def load_audio_mono_44k(audio_bytes: bytes, filename_hint: str = "input.wav") -> np.ndarray:
    """Decode any audio bytes (WAV, WebM, Ogg, MP3, M4A) into a 1D float64 numpy array at 44,100 Hz."""
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float64", always_2d=True)
        mono = np.mean(data, axis=1)
        if sr != SAMPLE_RATE:
            mono = resample_audio(mono, sr, SAMPLE_RATE)
        mono = declip_vocal_waveform(mono, SAMPLE_RATE)
        return soft_limit_audio(mono, 0.58)
    except Exception:
        pass

    suffix = Path(filename_hint).suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as src_tmp, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as dst_tmp:
        src_tmp.write(audio_bytes)
        src_tmp.flush()
        cmd = [
            "ffmpeg", "-y", "-i", src_tmp.name,
            "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-sample_fmt", "s16", dst_tmp.name,
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        data, sr = sf.read(dst_tmp.name, dtype="float64", always_2d=True)
        mono = np.mean(data, axis=1)
        mono = declip_vocal_waveform(mono, SAMPLE_RATE)
        return soft_limit_audio(mono, 0.58)


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or len(audio) == 0:
        return audio.astype(np.float64)
    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    return signal.resample_poly(audio, up, down).astype(np.float64)


def save_utau_wav(path: Path, audio: np.ndarray) -> None:
    """Save strictly as 44,100 Hz, 16-bit PCM, Mono WAV with clean studio headroom."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = np.clip(soft_limit_audio(audio, 0.58), -0.95, 0.95)
    sf.write(str(path), clean, SAMPLE_RATE, subtype="PCM_16")


def analyze_user_voice(audio: np.ndarray, sr: int = SAMPLE_RATE) -> Dict[str, Any]:
    """Extract the user's vocal characteristics from their recorded speech."""
    duration_sec = float(len(audio)) / sr
    if len(audio) < sr // 4:
        base_hz, base_note, semitone_shift = choose_user_base_note(175.0)
        return {
            "duration_sec": round(duration_sec, 2),
            "median_f0_hz": 175.0,
            "base_note_hz": base_hz,
            "base_note_name": base_note,
            "semitone_shift": semitone_shift,
            "formant_scale": 1.0,
            "brightness": 0.45,
        }

    voiced_f0 = np.array([], dtype=np.float64)
    if HAS_PARSELMOUTH and parselmouth is not None:
        try:
            snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=float(sr))
            pitch_obj = snd.to_pitch(time_step=0.010, pitch_floor=60.0, pitch_ceiling=480.0)
            f0_vals = pitch_obj.selected_array["frequency"]
            voiced_f0 = f0_vals[f0_vals > 62.0]
        except Exception:
            voiced_f0 = np.array([], dtype=np.float64)

    if len(voiced_f0) == 0 and HAS_PYWORLD and pw is not None:
        _f0, t = pw.dio(audio.astype(np.float64), sr, f0_floor=60.0, f0_ceil=500.0, frame_period=5.0)
        f0 = pw.stonemask(audio.astype(np.float64), _f0, t, sr)
        rms_frames = _frame_rms(audio, sr, frame_period=5.0, n_frames=len(f0))
        voiced_f0 = f0[(f0 > 62.0) & (rms_frames > np.percentile(rms_frames, 25))]
    elif len(voiced_f0) == 0:
        voiced_f0 = _estimate_f0_autocorr(audio, sr)

    if len(voiced_f0) > 0:
        median_f0 = float(np.median(voiced_f0))
        p10_f0 = float(np.percentile(voiced_f0, 10))
        p90_f0 = float(np.percentile(voiced_f0, 90))
    else:
        median_f0 = 165.0
        p10_f0 = 130.0
        p90_f0 = 210.0

    base_hz, base_note, semitone_shift = choose_user_base_note(median_f0)
    formant_scale = float(np.clip((median_f0 / 175.0) ** 0.20, 0.82, 1.20))

    n_fft = 2048
    freqs, psd = signal.welch(audio, fs=sr, nperseg=min(n_fft, len(audio)))
    psd = psd + 1e-12
    low_band = np.mean(psd[(freqs >= 100) & (freqs < 1000)])
    high_band = np.mean(psd[(freqs >= 1000) & (freqs < 4500)])
    brightness = float(np.clip(high_band / (low_band + high_band), 0.12, 0.85))

    return {
        "duration_sec": round(duration_sec, 2),
        "median_f0_hz": round(median_f0, 1),
        "f0_range_hz": [round(p10_f0, 1), round(p90_f0, 1)],
        "base_note_hz": round(base_hz, 2),
        "base_note_name": base_note,
        "semitone_shift": semitone_shift,
        "formant_scale": round(formant_scale, 3),
        "brightness": round(brightness, 3),
    }


def _frame_rms(audio: np.ndarray, sr: int, frame_period: float, n_frames: int) -> np.ndarray:
    hop = int((frame_period / 1000.0) * sr)
    win = hop * 2
    out = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        s = i * hop
        e = min(len(audio), s + win)
        if e > s:
            out[i] = np.sqrt(np.mean(audio[s:e] ** 2))
    return out


def _estimate_f0_autocorr(audio: np.ndarray, sr: int) -> np.ndarray:
    hop = int(0.01 * sr)
    win = int(0.04 * sr)
    min_lag = int(sr / 500.0)
    max_lag = int(sr / 65.0)
    f0s = []
    for start in range(0, len(audio) - win, hop):
        frame = audio[start : start + win]
        if np.sqrt(np.mean(frame ** 2)) < 0.03:
            continue
        frame = frame * np.hanning(len(frame))
        corr = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
        if max_lag < len(corr):
            sub = corr[min_lag:max_lag]
            if len(sub) > 0:
                peak_idx = int(np.argmax(sub)) + min_lag
                if corr[peak_idx] > 0.35 * corr[0]:
                    f0s.append(sr / float(peak_idx))
    return np.array(f0s, dtype=np.float64)


def stretch_log_spectrum_smooth(sp_chunk: np.ndarray, target_frames: int) -> np.ndarray:
    n_src = len(sp_chunk)
    if n_src == 0:
        return sp_chunk
    if n_src == 1 or target_frames <= 1:
        return np.tile(sp_chunk[0], (max(1, target_frames), 1))
    if n_src == target_frames:
        return sp_chunk.copy()

    src_t = np.linspace(0.0, 1.0, n_src)
    dst_t = np.linspace(0.0, 1.0, target_frames)
    log_sp = np.log(np.maximum(sp_chunk, 1e-10))
    f_interp = interp.interp1d(src_t, log_sp, axis=0, kind="linear", assume_sorted=True)
    return np.exp(f_interp(dst_t))


class UserVoiceCorpus:
    """
    Word-Aligned Time-Domain User Voice Extractor (Vosk Forced Alignment + WSOLA + Praat TD-PSOLA).
    1. Aligns every word in the user's recording (`reference.wav`) using Vosk grammar alignment.
    2. For each phoneme/diphone in `PHONEME_INVENTORY`, slices the ACTUAL word spoken by the user
       (e.g. "Daisy", "Bell", "give", "me", "half", "crazy", "all", "for", "the", "love", "of", "you")
       so the user's real regional/British accent, real vowel formants, and real consonants are
       100% preserved.
    3. Expands the interior vowel in the TIME DOMAIN via Phase-Locked WSOLA and gently centers pitch
       using Praat TD-PSOLA (`parselmouth`)—with ZERO spectral vocoder (`pyworld`) buzz!
    """

    def __init__(self, reference_audio: np.ndarray, voice_profile: Dict[str, Any], sr: int = SAMPLE_RATE):
        self.sr = sr
        self.voice_profile = voice_profile
        self.base_hz = float(voice_profile.get("base_note_hz", 110.0))
        self.formant_scale = float(voice_profile.get("formant_scale", 1.0))
        self.frame_period = 5.0

        self.audio = soft_limit_audio(declip_vocal_waveform(reference_audio, sr), 0.58)
        self.aligned_words: Dict[str, List[Tuple[float, float]]] = {}
        self.fallback_syllables: List[np.ndarray] = []

        self._align_spoken_words()
        self._extract_fallback_syllables()

    def _align_spoken_words(self) -> None:
        model = get_vosk_model()
        if model is None or KaldiRecognizer is None or len(self.audio) < self.sr // 3:
            return

        try:
            x16 = resample_audio(self.audio, self.sr, 16000)
            pcm16 = (np.clip(x16, -0.99, 0.99) * 32767.0).astype(np.int16).tobytes()

            rec = KaldiRecognizer(model, 16000, json.dumps(ALIGNMENT_VOCABULARY))
            rec.SetWords(True)
            rec.AcceptWaveform(pcm16)
            res = json.loads(rec.FinalResult())
            words = res.get("result", [])

            for w in words:
                word_str = str(w.get("word", "")).strip().lower()
                st = float(w.get("start", 0.0))
                en = float(w.get("end", 0.0))
                if word_str and word_str != "[unk]" and en - st >= 0.05:
                    self.aligned_words.setdefault(word_str, []).append((st, en))
        except Exception:
            pass

    def _extract_fallback_syllables(self) -> None:
        """Extract clean single-word voiced slices as fallbacks if a user said a custom sentence."""
        hop = int(0.010 * self.sr)
        win = int(0.025 * self.sr)
        if len(self.audio) < win * 3:
            return
        rms = np.array([
            np.sqrt(np.mean(self.audio[i : i + win] ** 2))
            for i in range(0, len(self.audio) - win, hop)
        ])
        thresh = max(0.025, float(np.percentile(rms, 35)))
        active = rms > thresh
        in_seg = False
        st = 0
        for i, val in enumerate(active):
            if val and not in_seg:
                in_seg = True
                st = i
            elif not val and in_seg:
                dur_s = (i - st) * 0.010
                if 0.12 <= dur_s <= 0.65:
                    s_idx = max(0, st * hop)
                    e_idx = min(len(self.audio), i * hop + win)
                    self.fallback_syllables.append(self.audio[s_idx:e_idx])
                in_seg = False

    def _find_best_word_slice(self, phoneme_item: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Look up the user's actual spoken word for `phoneme_item` from Vosk alignment
        and slice the requested syllable/sub-word region.
        """
        candidates: List[Tuple[str, str]] = phoneme_item.get("source_words", [])
        for word_key, slice_mode in candidates:
            spans = self.aligned_words.get(word_key.lower())
            if not spans:
                continue
            st_sec, en_sec = max(spans, key=lambda sp: sp[1] - sp[0])
            sliced = self._slice_word_region(st_sec, en_sec, slice_mode)
            if sliced is not None and len(sliced) >= int(0.075 * self.sr):
                return sliced
        return None

    def _slice_word_region(self, st_sec: float, en_sec: float, slice_mode: str) -> Optional[np.ndarray]:
        pad_pre = 0.012
        pad_post = 0.018
        s_idx = max(0, int((st_sec - pad_pre) * self.sr))
        e_idx = min(len(self.audio), int((en_sec + pad_post) * self.sr))
        word_wav = self.audio[s_idx:e_idx]
        n = len(word_wav)
        if n < int(0.07 * self.sr):
            return None

        if slice_mode == "full":
            return word_wav

        if slice_mode in ("pre_sibilant", "from_sibilant"):
            b_hi, a_hi = signal.butter(2, [3100 / (self.sr / 2), 8500 / (self.sr / 2)], btype="band")
            b_lo, a_lo = signal.butter(2, [180 / (self.sr / 2), 1800 / (self.sr / 2)], btype="band")
            hi = signal.lfilter(b_hi, a_hi, word_wav) ** 2
            lo = signal.lfilter(b_lo, a_lo, word_wav) ** 2 + 1e-6
            win = max(16, int(0.014 * self.sr))
            ratio = np.convolve(hi, np.ones(win) / win, mode="same") / np.convolve(
                lo, np.ones(win) / win, mode="same"
            )
            search_s = int(n * 0.25)
            search_e = int(n * 0.82)
            if search_e > search_s + 16:
                z_peak = search_s + int(np.argmax(ratio[search_s:search_e]))
                z_onset = max(int(n * 0.28), z_peak - int(0.028 * self.sr))
            else:
                z_onset = n // 2

            if slice_mode == "pre_sibilant":
                return word_wav[: min(n, z_onset + int(0.015 * self.sr))]
            else:
                return word_wav[max(0, z_onset - int(0.012 * self.sr)) :]

        if slice_mode == "first_half":
            return word_wav[: int(n * 0.58)]

        if slice_mode == "second_half":
            return word_wav[int(n * 0.40) :]

        if slice_mode == "vowel_core":
            b_lo, a_lo = signal.butter(2, [160 / (self.sr / 2), 2400 / (self.sr / 2)], btype="band")
            v_energy = signal.lfilter(b_lo, a_lo, word_wav) ** 2
            win = max(16, int(0.015 * self.sr))
            env = np.convolve(v_energy, np.ones(win) / win, mode="same")
            peak_e = float(np.max(env))
            if peak_e > 1e-7:
                voiced_idx = np.where(env > peak_e * 0.22)[0]
                if len(voiced_idx) >= int(0.06 * self.sr):
                    vs = int(voiced_idx[0]) + int(len(voiced_idx) * 0.12)
                    ve = int(voiced_idx[-1]) - int(len(voiced_idx) * 0.12)
                    if ve - vs >= int(0.06 * self.sr):
                        return word_wav[vs:ve]
            return word_wav[int(n * 0.18) : int(n * 0.84)]

        return word_wav

    def get_raw_word_slice(self, phoneme_item: Dict[str, Any]) -> np.ndarray:
        """
        Return the untouched, un-vocoded microphone word slice for `phoneme_item`
        (with gentle attack/release fades so single-pass TD-PSOLA resamplers have zero clicks).
        """
        raw_slice = self._find_best_word_slice(phoneme_item)
        if raw_slice is None:
            if self.fallback_syllables:
                idx = abs(hash(phoneme_item["id"])) % len(self.fallback_syllables)
                raw_slice = self.fallback_syllables[idx]
            else:
                return synthesize_local_phoneme(phoneme_item, self.voice_profile, self.sr, self.base_hz)

        x = trim_silence(soft_limit_audio(raw_slice, 0.58), self.sr)
        attack = int(0.012 * self.sr)
        release = int(0.022 * self.sr)
        if len(x) > attack + release:
            x = x.copy()
            x[:attack] *= np.linspace(0.0, 1.0, attack)
            x[-release:] *= np.linspace(1.0, 0.0, release)
        return soft_limit_audio(x, 0.58)

    def synthesize_phoneme_from_user_voice(self, phoneme_item: Dict[str, Any]) -> np.ndarray:
        """
        Prepare a ~440ms natural OpenUTAU voicebank `.wav` sample directly in the time domain:
        - Uses Phase-Locked Time-Domain WSOLA (`wsola_stretch_phase_locked`) to expand short spoken vowels
          without any FFT vocoder (`pyworld`).
        - Uses Praat TD-PSOLA (`psola_pitch_transfer`) with speaker-locked F0 floor/ceiling to gently
          center the pitch around `self.base_hz` while preserving 45% of the speaker's natural pitch contour!
        """
        raw_word = self.get_raw_word_slice(phoneme_item)
        ptype = phoneme_item.get("type", "cv")
        min_head = int(0.022 * self.sr) if ptype == "vowel" else int(0.055 * self.sr)

        # Expand in the time domain to ~430ms for OpenUTAU sustain stability
        target_len = int(0.43 * self.sr)
        expanded = wsola_stretch_phase_locked(raw_word, target_len, self.sr, min_head_samples=min_head)

        base_hz = self.base_hz
        f0_floor = max(55.0, base_hz * 0.60)
        f0_ceil = min(380.0, base_hz * 1.65)

        def _gentle_contour(times: np.ndarray, orig_f0: np.ndarray, med_f0: float) -> np.ndarray:
            ratio = np.where(orig_f0 > f0_floor, orig_f0 / max(60.0, med_f0), 1.0)
            # Retain 45% of the speaker's real pitch inflection around their natural base note
            natural_inflection = np.clip(ratio ** 0.45, 0.92, 1.08)
            return base_hz * natural_inflection

        wav_out = psola_pitch_transfer(expanded, self.sr, _gentle_contour, f0_floor=f0_floor, f0_ceil=f0_ceil)

        attack = int(0.014 * self.sr)
        release = int(0.028 * self.sr)
        if len(wav_out) > attack + release:
            wav_out[:attack] *= np.linspace(0.0, 1.0, attack)
            wav_out[-release:] *= np.linspace(1.0, 0.0, release)

        return soft_limit_audio(wav_out, 0.58)


def slice_batch_with_timestamps(
    batch_audio: np.ndarray,
    alignment: Optional[Dict[str, Any]],
    batch_items: List[Dict[str, Any]],
    sr: int = SAMPLE_RATE,
) -> List[np.ndarray]:
    n_expected = len(batch_items)
    if alignment and "characters" in alignment and "character_start_times_seconds" in alignment:
        chars: List[str] = alignment["characters"]
        starts: List[float] = alignment["character_start_times_seconds"]
        ends: List[float] = alignment["character_end_times_seconds"]

        word_spans: List[Tuple[float, float]] = []
        cur_start: Optional[float] = None
        cur_end: Optional[float] = None

        for ch, st, en in zip(chars, starts, ends):
            if ch.isalpha():
                if cur_start is None:
                    cur_start = float(st)
                cur_end = float(en)
            elif ch in (".", ";", ",", "|", "\n"):
                if cur_start is not None and cur_end is not None:
                    word_spans.append((cur_start, cur_end))
                    cur_start = None
                    cur_end = None
        if cur_start is not None and cur_end is not None:
            word_spans.append((cur_start, cur_end))

        if len(word_spans) == n_expected:
            clips = []
            for st, en in word_spans:
                pad_pre = 0.035
                pad_post = 0.045
                s_idx = max(0, int((st - pad_pre) * sr))
                e_idx = min(len(batch_audio), int((en + pad_post) * sr))
                clip = batch_audio[s_idx:e_idx]
                clips.append(trim_silence(clip, sr))
            return clips

    return _slice_by_energy_islands(batch_audio, n_expected, sr)


def _slice_by_energy_islands(audio: np.ndarray, n_expected: int, sr: int = SAMPLE_RATE) -> List[np.ndarray]:
    frame_len = int(0.015 * sr)
    hop = int(0.005 * sr)
    if len(audio) < frame_len * 2:
        return [audio for _ in range(n_expected)]

    rms = np.array([
        np.sqrt(np.mean(audio[i : i + frame_len] ** 2))
        for i in range(0, len(audio) - frame_len, hop)
    ])
    thresh = max(0.018, float(np.percentile(rms, 30)) * 1.4)
    active = rms > thresh

    islands: List[Tuple[int, int]] = []
    in_island = False
    start_f = 0
    for idx, val in enumerate(active):
        if val and not in_island:
            in_island = True
            start_f = idx
        elif not val and in_island:
            if (idx - start_f) * hop >= int(0.06 * sr):
                islands.append((start_f * hop, idx * hop + frame_len))
            in_island = False
    if in_island:
        islands.append((start_f * hop, len(audio)))

    if len(islands) == n_expected:
        return [trim_silence(audio[s:e], sr) for s, e in islands]

    step = len(audio) // max(1, n_expected)
    return [trim_silence(audio[i * step : (i + 1) * step], sr) for i in range(n_expected)]


def trim_silence(audio: np.ndarray, sr: int = SAMPLE_RATE, thresh_ratio: float = 0.04) -> np.ndarray:
    if len(audio) == 0:
        return audio
    peak = np.max(np.abs(audio))
    if peak < 1e-5:
        return audio
    win = int(0.008 * sr)
    if len(audio) <= win * 2:
        return audio
    env = np.convolve(np.abs(audio), np.ones(win) / win, mode="same")
    active = np.where(env > peak * thresh_ratio)[0]
    if len(active) == 0:
        return audio
    s = max(0, int(active[0]) - int(0.015 * sr))
    e = min(len(audio), int(active[-1]) + int(0.020 * sr))
    return audio[s:e]


def flatten_pitch_and_prepare_sample(
    raw_clip: np.ndarray,
    phoneme_item: Dict[str, Any],
    target_f0: float = TARGET_NOTE_HZ,
    sr: int = SAMPLE_RATE,
    already_flattened: bool = False,
) -> Tuple[np.ndarray, float]:
    clip = trim_silence(raw_clip, sr)
    ptype = phoneme_item.get("type", "cv")

    if already_flattened:
        synthesized = soft_limit_audio(clip, 0.58)
        consonant_dur_ms = 18.0 if ptype == "vowel" else 65.0
    elif len(clip) >= int(0.075 * sr):
        min_head = int(0.022 * sr) if ptype == "vowel" else int(0.055 * sr)
        expanded = wsola_stretch_phase_locked(clip, int(0.44 * sr), sr, min_head_samples=min_head)
        f0_floor = max(55.0, target_f0 * 0.60)
        f0_ceil = min(380.0, target_f0 * 1.65)

        def _contour(times: np.ndarray, orig_f0: np.ndarray, med_f0: float) -> np.ndarray:
            ratio = np.where(orig_f0 > f0_floor, orig_f0 / max(60.0, med_f0), 1.0)
            return target_f0 * np.clip(ratio ** 0.40, 0.93, 1.07)

        synthesized = psola_pitch_transfer(expanded, sr, _contour, f0_floor=f0_floor, f0_ceil=f0_ceil)
        consonant_dur_ms = 18.0 if ptype == "vowel" else 65.0
    else:
        synthesized = soft_limit_audio(clip, 0.58)
        consonant_dur_ms = 18.0 if ptype == "vowel" else 65.0

    fade_samples = int(0.014 * sr)
    if len(synthesized) > fade_samples * 2:
        synthesized[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples)
        synthesized[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples)

    synthesized = soft_limit_audio(synthesized, 0.58)
    pad_len = int(0.075 * sr)
    padded = np.concatenate([
        np.zeros(pad_len, dtype=np.float64),
        synthesized,
        np.zeros(pad_len, dtype=np.float64),
    ])
    return padded, consonant_dur_ms


def compute_oto_parameters(
    sample_audio: np.ndarray,
    phoneme_item: Dict[str, Any],
    consonant_dur_ms: float,
    sr: int = SAMPLE_RATE,
) -> Dict[str, float]:
    total_ms = (len(sample_audio) / float(sr)) * 1000.0
    ptype = phoneme_item.get("type", "cv")

    offset = 65.0
    if ptype == "vowel":
        preutterance = 25.0
        overlap = 15.0
        consonant = 80.0
    elif ptype == "vc":
        preutterance = 35.0
        overlap = 20.0
        consonant = 105.0
    else:
        preutterance = round(float(np.clip(consonant_dur_ms + 10.0, 45.0, 115.0)), 1)
        overlap = round(max(18.0, preutterance * 0.42), 1)
        consonant = round(preutterance + 95.0, 1)

    cutoff = -90.0
    active_ms = total_ms - offset + cutoff
    if consonant >= active_ms - 40.0:
        consonant = max(60.0, round(active_ms * 0.50, 1))

    return {
        "offset": round(offset, 1),
        "consonant": round(consonant, 1),
        "cutoff": round(cutoff, 1),
        "preutterance": round(preutterance, 1),
        "overlap": round(overlap, 1),
        "duration_ms": round(total_ms, 1),
    }


def synthesize_local_phoneme(
    phoneme_item: Dict[str, Any],
    voice_profile: Dict[str, Any],
    sr: int = SAMPLE_RATE,
    target_f0: float = TARGET_NOTE_HZ,
) -> np.ndarray:
    base_hz = float(voice_profile.get("base_note_hz", target_f0))
    formant_scale = float(voice_profile.get("formant_scale", 1.0))
    f1, f2, f3 = phoneme_item.get("formants", (650, 1400, 2450))
    n_samples = int(0.52 * sr)
    t = np.arange(n_samples, dtype=np.float64) / sr
    glottal = np.zeros(n_samples, dtype=np.float64)
    for h in range(1, 20):
        glottal += (1.0 / (h ** 1.15)) * np.sin(2.0 * np.pi * base_hz * h * t)
    sig = np.zeros(n_samples, dtype=np.float64)
    for freq_hz, bw_hz, gain in [(f1 * formant_scale, 80.0, 1.0), (f2 * formant_scale, 110.0, 0.7), (f3 * formant_scale, 150.0, 0.35)]:
        fc = float(np.clip(freq_hz, 180.0, sr * 0.45))
        b, a = signal.iirpeak(fc / (sr / 2.0), max(2.0, fc / bw_hz))
        sig += gain * signal.lfilter(b, a, glottal)
    return soft_limit_audio(sig, 0.80)
