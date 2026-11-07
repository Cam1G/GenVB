"""
Audio DSP & Vosk Forced-Alignment Pipeline for OpenUTAU Voicebank Generation:
- Uses Vosk grammar-guided speech recognition to locate the exact millisecond start/end
  timestamps of every word the user spoke in their recording.
- Slices the user's ACTUAL spoken words and syllables ("Daisy", "Bell", "give", "me", "half",
  "crazy", "all", "for", "the", "love", "of", "you", etc.) directly from their recording—
  preserving 100% of their real regional/British accent, real consonants, and real vocal tract!
- Smooth log-spectral time-interpolation on the interior vowel nucleus (zero 1-frame robotic
  freeze, and zero multi-word "wayo-wayo" ping-pong looping).
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.interpolate as interp
import scipy.signal as signal
import soundfile as sf

from .phonemes import ALIGNMENT_VOCABULARY

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


def choose_user_base_note(median_f0_hz: float) -> Tuple[float, str, int]:
    """
    Choose a natural UTAU base reference pitch close to the user's actual speaking pitch
    so the voicebank and Daisy Bell stay in their comfortable natural chest register
    (avoiding extreme upward pitch shifts that sound robotic or strained).
    Returns (base_hz, note_name, semitone_shift_from_c4).
    """
    if median_f0_hz < 125.0:
        # Low Male / Baritone (~85-124 Hz) -> Base note A2 (110.00 Hz, MIDI 45, -15 semitones from C4)
        return 110.00, "A2", -15
    elif median_f0_hz < 155.0:
        # Mid-Low / Tenor (~125-154 Hz) -> Base note C3 (130.81 Hz, MIDI 48, -12 semitones from C4)
        return 130.81, "C3", -12
    elif median_f0_hz < 195.0:
        # Alto / High Tenor (~155-194 Hz) -> Base note F3 (174.61 Hz, MIDI 53, -7 semitones from C4)
        return 174.61, "F3", -7
    elif median_f0_hz < 235.0:
        # Mezzo (~195-234 Hz) -> Base note A3 (220.00 Hz, MIDI 57, -3 semitones from C4)
        return 220.00, "A3", -3
    else:
        # Soprano (>= 235 Hz) -> Base note C4 (261.63 Hz, MIDI 60)
        return 261.63, "C4", 0


def soft_limit_audio(audio: np.ndarray, target_peak: float = 0.80) -> np.ndarray:
    """
    Transparent studio soft-knee saturation + peak normalization so high notes NEVER clip.
    """
    if len(audio) == 0:
        return audio
    x = audio - np.mean(audio)
    peak = float(np.max(np.abs(x)))
    if peak < 1e-6:
        return x.astype(np.float64)
    # Normalize to unit range, apply gentle soft-knee tanh compression on transient spikes, then scale
    norm = x / peak
    compressed = np.tanh(norm * 1.15) / np.tanh(1.15)
    return (compressed * target_peak).astype(np.float64)


def normalize_peak(audio: np.ndarray, target_peak: float = 0.82) -> np.ndarray:
    return soft_limit_audio(audio, target_peak=target_peak)


def load_audio_mono_44k(audio_bytes: bytes, filename_hint: str = "input.wav") -> np.ndarray:
    """Decode any audio bytes (WAV, WebM, Ogg, MP3, M4A) into a 1D float64 numpy array at 44,100 Hz."""
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float64", always_2d=True)
        mono = np.mean(data, axis=1)
        if sr != SAMPLE_RATE:
            mono = resample_audio(mono, sr, SAMPLE_RATE)
        return soft_limit_audio(mono, 0.85)
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
        return soft_limit_audio(mono, 0.85)


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or len(audio) == 0:
        return audio.astype(np.float64)
    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    return signal.resample_poly(audio, up, down).astype(np.float64)


def save_utau_wav(path: Path, audio: np.ndarray) -> None:
    """Save strictly as 44,100 Hz, 16-bit PCM, Mono WAV with soft-knee anti-clipping."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = np.clip(soft_limit_audio(audio, 0.80), -0.98, 0.98)
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

    if HAS_PYWORLD and pw is not None:
        _f0, t = pw.dio(audio.astype(np.float64), sr, f0_floor=60.0, f0_ceil=500.0, frame_period=5.0)
        f0 = pw.stonemask(audio.astype(np.float64), _f0, t, sr)
        rms_frames = _frame_rms(audio, sr, frame_period=5.0, n_frames=len(f0))
        voiced_f0 = f0[(f0 > 62.0) & (rms_frames > np.percentile(rms_frames, 25))]
    else:
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
    """
    Smoothly time-stretch a spectral matrix `sp_chunk` (shape [N, bins]) to `target_frames`
    using linear interpolation in log-spectral space along the time axis (forward-only,
    never ping-pong looping and never freezing a single 1D robot frame!).
    """
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
    Word-Aligned Direct User Voice Extractor (Vosk Forced Alignment + WORLD Natural Resynthesis).
    1. Aligns every word in the user's recording (`reference.wav`) using Vosk grammar alignment.
    2. For each phoneme/diphone in `PHONEME_INVENTORY`, slices the ACTUAL word spoken by the user
       (e.g. "Daisy", "Bell", "give", "me", "half", "crazy", "all", "for", "the", "love", "of", "you")
       so the user's real regional/British accent, real vowel formants, and real consonants are
       100% preserved without synthetic formant overwriting!
    3. Time-stretches only the pure interior vowel nucleus of that single isolated word via
       smooth forward log-spectral interpolation (no frozen 1-frame robot buzz, no multi-word "wayo").
    """

    def __init__(self, reference_audio: np.ndarray, voice_profile: Dict[str, Any], sr: int = SAMPLE_RATE):
        self.sr = sr
        self.voice_profile = voice_profile
        self.base_hz = float(voice_profile.get("base_note_hz", 110.0))
        self.formant_scale = float(voice_profile.get("formant_scale", 1.0))
        self.frame_period = 5.0

        self.audio = soft_limit_audio(reference_audio, 0.85)
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
            # Pick the longest, clearest occurrence of this word
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
            # Locate internal sibilant (-z- in 'daisy'/'crazy'/'lazy', -s- in 'answer'/'across'/'voices')
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
            # Trim outer consonants to isolate the pure spoken vowel in the user's accent
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

    def synthesize_phoneme_from_user_voice(self, phoneme_item: Dict[str, Any]) -> np.ndarray:
        """
        Extract the user's real spoken syllable/word for `phoneme_item` and prepare a
        480ms natural UTAU sample at `self.base_hz`:
        - Preserves 100% of the user's actual spectral envelope `sp` (their real accent & formants!)
        - Preserves their real spoken consonant onset
        - Preserves 25% of their natural human vocal-cord micro-intonation (never robotic!)
        - Smoothly stretches only the pure interior vowel nucleus via forward log-spectral interpolation
        """
        raw_slice = self._find_best_word_slice(phoneme_item)
        if raw_slice is None:
            if self.fallback_syllables:
                idx = abs(hash(phoneme_item["id"])) % len(self.fallback_syllables)
                raw_slice = self.fallback_syllables[idx]
            else:
                return synthesize_local_phoneme(phoneme_item, self.voice_profile, self.sr, self.base_hz)

        if not (HAS_PYWORLD and pw is not None) or len(raw_slice) < int(0.065 * self.sr):
            return soft_limit_audio(raw_slice, 0.80)

        x = soft_limit_audio(raw_slice, 0.82).astype(np.float64)
        ptype = phoneme_item.get("type", "cv")

        _f0, t = pw.dio(x, self.sr, f0_floor=58.0, f0_ceil=480.0, frame_period=self.frame_period)
        f0 = pw.stonemask(x, _f0, t, self.sr)
        n_frames = len(f0)
        if n_frames < 8:
            return x

        voiced_vals = f0[f0 > 60.0]
        med_f0 = float(np.median(voiced_vals)) if len(voiced_vals) > 0 else self.base_hz

        # Identify consonant onset vs voiced vowel body
        cons_frames = 2 if ptype == "vowel" else min(n_frames // 3, 14)
        voiced_mask = f0 > 60.0
        if n_frames > cons_frames + 3:
            voiced_mask[cons_frames:] = True

        # Ensure non-zero F0 on all vowel frames BEFORE calling cheaptrick & d4c
        f0_for_sp = np.where(voiced_mask, np.where(f0 > 60.0, f0, med_f0), 0.0)

        sp = np.maximum(pw.cheaptrick(x, f0_for_sp, t, self.sr), 1e-10)
        ap = np.clip(pw.d4c(x, f0_for_sp, t, self.sr), 0.001, 0.85)
        # Keep natural human breath warmth on the vowel while capping excess noise <= 0.25
        ap[cons_frames:] = np.minimum(ap[cons_frames:], 0.25)

        # Preserve 25% of the user's real vocal-cord micro-prosody around `self.base_hz`
        # so the sample sounds like a real human voice and not a flat digital buzzer!
        raw_ratio = np.where(f0 > 60.0, f0 / med_f0, 1.0)
        natural_micro_contour = np.clip(raw_ratio ** 0.25, 0.96, 1.04)

        # Time-stretch ONLY the interior 55% of this word's vowel nucleus to reach ~98 frames (490ms)
        target_total_frames = 98
        if n_frames < target_total_frames and n_frames >= cons_frames + 6:
            v_len = n_frames - cons_frames
            nuc_start = cons_frames + max(2, int(v_len * 0.20))
            nuc_end = cons_frames + max(nuc_start - cons_frames + 3, int(v_len * 0.78))
            nuc_end = min(n_frames - 2, nuc_end)

            head_sp = sp[:nuc_start]
            core_sp = sp[nuc_start:nuc_end]
            tail_sp = sp[nuc_end:]

            head_ap = ap[:nuc_start]
            core_ap = ap[nuc_start:nuc_end]
            tail_ap = ap[nuc_end:]

            head_mc = natural_micro_contour[:nuc_start]
            core_mc = natural_micro_contour[nuc_start:nuc_end]
            tail_mc = natural_micro_contour[nuc_end:]

            needed_core = max(len(core_sp), target_total_frames - len(head_sp) - len(tail_sp))
            stretched_sp = stretch_log_spectrum_smooth(core_sp, needed_core)
            stretched_ap = stretch_log_spectrum_smooth(core_ap, needed_core)

            mc_src_t = np.linspace(0.0, 1.0, len(core_mc))
            mc_dst_t = np.linspace(0.0, 1.0, needed_core)
            stretched_mc = np.interp(mc_dst_t, mc_src_t, core_mc)

            sp_out = np.vstack([head_sp, stretched_sp, tail_sp])
            ap_out = np.vstack([head_ap, stretched_ap, tail_ap])
            mc_out = np.concatenate([head_mc, stretched_mc, tail_mc])
        else:
            sp_out = sp
            ap_out = ap
            mc_out = natural_micro_contour

        n_out = len(sp_out)
        voiced_out = np.ones(n_out, dtype=bool)
        if cons_frames > 0:
            voiced_out[:cons_frames] = voiced_mask[:cons_frames]
        f0_out = np.where(voiced_out, self.base_hz * mc_out, 0.0)

        wav_out = pw.synthesize(
            np.ascontiguousarray(f0_out, dtype=np.float64),
            np.ascontiguousarray(np.maximum(sp_out, 1e-10), dtype=np.float64),
            np.ascontiguousarray(np.clip(ap_out, 0.001, 0.85), dtype=np.float64),
            self.sr,
            frame_period=self.frame_period,
        )
        wav_out = np.nan_to_num(wav_out, nan=0.0, posinf=0.0, neginf=0.0)

        # Splice the user's raw un-vocoded consonant onset waveform at the very start for 100% real consonant crispness!
        hop_samples = int((self.frame_period / 1000.0) * self.sr)
        cons_samples = cons_frames * hop_samples
        if ptype != "vowel" and cons_samples > 32 and len(x) > cons_samples + 32 and len(wav_out) > cons_samples + 32:
            xfade = min(cons_samples // 2, int(0.015 * self.sr))
            raw_cons = x[:cons_samples]
            fade_o = np.linspace(1.0, 0.0, xfade)
            fade_i = np.linspace(0.0, 1.0, xfade)
            wav_out[: cons_samples - xfade] = raw_cons[: cons_samples - xfade]
            wav_out[cons_samples - xfade : cons_samples] = (
                raw_cons[cons_samples - xfade : cons_samples] * fade_o
                + wav_out[cons_samples - xfade : cons_samples] * fade_i
            )

        attack = int(0.012 * self.sr)
        release = int(0.028 * self.sr)
        if len(wav_out) > attack + release:
            wav_out[:attack] *= np.linspace(0.0, 1.0, attack)
            wav_out[-release:] *= np.linspace(1.0, 0.0, release)

        return soft_limit_audio(wav_out, 0.80)


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
        synthesized = soft_limit_audio(clip, 0.80)
        consonant_dur_ms = 18.0 if ptype == "vowel" else 65.0
    elif HAS_PYWORLD and pw is not None and len(clip) >= int(0.08 * sr):
        x = soft_limit_audio(clip, 0.82).astype(np.float64)
        _f0, t = pw.dio(x, sr, f0_floor=58.0, f0_ceil=520.0, frame_period=5.0)
        f0 = pw.stonemask(x, _f0, t, sr)

        voiced_mask = f0 > 60.0
        cons_frames = 3 if ptype == "vowel" else 13
        if len(voiced_mask) > cons_frames + 4:
            voiced_mask[cons_frames:] = True

        voiced_f0_vals = f0[f0 > 60.0]
        med_f0 = float(np.median(voiced_f0_vals)) if len(voiced_f0_vals) > 0 else target_f0
        f0_clean = np.where(voiced_mask, np.where(f0 > 60.0, f0, med_f0), 0.0)

        sp = np.maximum(pw.cheaptrick(x, f0_clean, t, sr), 1e-10)
        ap = np.clip(pw.d4c(x, f0_clean, t, sr), 0.001, 0.85)
        ap[voiced_mask] = np.minimum(ap[voiced_mask], 0.22)

        voiced_indices = np.where(voiced_mask)[0]
        first_voiced_frame = int(voiced_indices[0]) if len(voiced_indices) > 0 else 0

        raw_ratio = np.where(f0 > 60.0, f0 / med_f0, 1.0)
        micro_contour = np.clip(raw_ratio ** 0.25, 0.96, 1.04)
        flat_f0 = np.where(voiced_mask, target_f0 * micro_contour, 0.0)

        n_frames = len(flat_f0)
        target_min_frames = 100
        if n_frames < target_min_frames and len(voiced_indices) >= 6:
            mid_s = int(voiced_indices[len(voiced_indices) // 4])
            mid_e = int(voiced_indices[(3 * len(voiced_indices)) // 4])
            if mid_e <= mid_s + 2:
                mid_e = min(n_frames, mid_s + 4)
            needed = target_min_frames - (n_frames - (mid_e - mid_s))
            stretched_sp = stretch_log_spectrum_smooth(sp[mid_s:mid_e], needed)
            stretched_ap = stretch_log_spectrum_smooth(ap[mid_s:mid_e], needed)
            f0_mid = np.full(needed, target_f0, dtype=np.float64)

            flat_f0 = np.concatenate([flat_f0[:mid_s], f0_mid, flat_f0[mid_e:]])
            sp = np.vstack([sp[:mid_s], stretched_sp, sp[mid_e:]])
            ap = np.vstack([ap[:mid_s], stretched_ap, ap[mid_e:]])

        synthesized = pw.synthesize(
            np.ascontiguousarray(flat_f0, dtype=np.float64),
            np.ascontiguousarray(sp, dtype=np.float64),
            np.ascontiguousarray(ap, dtype=np.float64),
            sr,
            frame_period=5.0,
        )
        synthesized = np.nan_to_num(synthesized, nan=0.0, posinf=0.0, neginf=0.0)
        consonant_dur_ms = float(first_voiced_frame * 5.0)
        if ptype == "vowel":
            consonant_dur_ms = min(consonant_dur_ms, 20.0)
        else:
            consonant_dur_ms = float(np.clip(consonant_dur_ms, 40.0, 120.0))
    else:
        synthesized = soft_limit_audio(clip, 0.80)
        consonant_dur_ms = 18.0 if ptype == "vowel" else 65.0

    fade_samples = int(0.015 * sr)
    if len(synthesized) > fade_samples * 2:
        synthesized[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples)
        synthesized[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples)

    synthesized = soft_limit_audio(synthesized, 0.80)
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
