"""
UTAU Resampler Engine & Project Exporter for "Daisy Bell (Bicycle Built for Two)" (1892 / IBM 7094 1961).
- Automatically transposes the melody into the user's natural vocal register (A2, C3, F3, A3, or C4).
- Uses Smooth Log-Spectral Vowel Interpolation (preserving 100% of the user's real Vosk-aligned
  word formants and regional/British accent without robotic 1-frame freezing or "wayo" loops!).
- Includes Equal-Loudness High-Note Compensation + Studio Soft-Knee Limiting for zero clipping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.signal as signal
import soundfile as sf
import yaml

from .audio_processor import (
    HAS_PYWORLD,
    SAMPLE_RATE,
    TARGET_NOTE_HZ,
    pw,
    save_utau_wav,
    soft_limit_audio,
    stretch_log_spectrum_smooth,
)

BPM = 138.0  # Natural, expressive 3/4 waltz tempo
BEAT_SEC = 60.0 / BPM

# "Daisy Bell" Chorus Score (Reference Key: C Major, automatically transposed by user's semitone_shift)
DAISY_BELL_NOTES: List[Dict[str, Any]] = [
    # Phrase 1: "Dai - sy, Dai - sy,"
    {"lyric": "Dai-", "alias": "d ey", "sample_id": "16_d_ey", "midi": 72, "start_beat": 0.0, "dur_beats": 3.0},
    {"lyric": "-sy,", "alias": "z iy", "sample_id": "17_z_iy", "midi": 67, "start_beat": 3.0, "dur_beats": 3.0},
    {"lyric": "Dai-", "alias": "d ey", "sample_id": "16_d_ey", "midi": 64, "start_beat": 6.0, "dur_beats": 3.0},
    {"lyric": "-sy,", "alias": "z iy", "sample_id": "17_z_iy", "midi": 60, "start_beat": 9.0, "dur_beats": 3.0},

    # Phrase 2: "give me your an - swer do!"
    {"lyric": "give", "alias": "g ih", "sample_id": "18_g_ih", "midi": 62, "start_beat": 12.0, "dur_beats": 1.0},
    {"lyric": "me", "alias": "m iy", "sample_id": "19_m_iy", "midi": 64, "start_beat": 13.0, "dur_beats": 1.0},
    {"lyric": "your", "alias": "y ao", "sample_id": "20_y_ao", "midi": 65, "start_beat": 14.0, "dur_beats": 1.0},
    {"lyric": "an-", "alias": "- ae n", "sample_id": "21_ae_n", "midi": 62, "start_beat": 15.0, "dur_beats": 2.0},
    {"lyric": "-swer", "alias": "s er", "sample_id": "22_s_er", "midi": 65, "start_beat": 17.0, "dur_beats": 1.0},
    {"lyric": "do!", "alias": "d uw", "sample_id": "23_d_uw", "midi": 60, "start_beat": 18.0, "dur_beats": 5.0},

    # Phrase 3: "I'm half cra - zy,"
    {"lyric": "I'm", "alias": "ay m", "sample_id": "06_ay", "midi": 69, "start_beat": 24.0, "dur_beats": 3.0},
    {"lyric": "half", "alias": "hh ae", "sample_id": "24_hh_ae", "midi": 74, "start_beat": 27.0, "dur_beats": 3.0},
    {"lyric": "cra-", "alias": "k r", "sample_id": "25_k_r_ey", "midi": 71, "start_beat": 30.0, "dur_beats": 3.0},
    {"lyric": "-zy,", "alias": "z iy", "sample_id": "17_z_iy", "midi": 67, "start_beat": 33.0, "dur_beats": 3.0},

    # Phrase 4: "all for the love of you!"
    {"lyric": "all", "alias": "ao l", "sample_id": "48_ao_l", "midi": 64, "start_beat": 36.0, "dur_beats": 1.0},
    {"lyric": "for", "alias": "f ao", "sample_id": "26_f_ao", "midi": 66, "start_beat": 37.0, "dur_beats": 1.0},
    {"lyric": "the", "alias": "dh ah", "sample_id": "27_dh_ah", "midi": 67, "start_beat": 38.0, "dur_beats": 1.0},
    {"lyric": "love", "alias": "l ah", "sample_id": "28_l_ah", "midi": 69, "start_beat": 39.0, "dur_beats": 2.0},
    {"lyric": "of", "alias": "v ah", "sample_id": "29_v_ah", "midi": 71, "start_beat": 41.0, "dur_beats": 1.0},
    {
        "lyric": "you!",
        "alias": "y uw",
        "sample_id": "30_y_uw",
        "midi": 69,
        "slur_to_midi": 67,
        "slur_at_beat_offset": 3.0,
        "start_beat": 42.0,
        "dur_beats": 5.5,
    },
]


def midi_to_hz(midi_note: int) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))


def render_daisy_bell_preview(
    voicebank_dir: Path,
    samples_meta: List[Dict[str, Any]],
    output_wav_path: Path,
    voice_profile: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Synthesize "Daisy Bell" using the newly generated UTAU voicebank samples and oto.ini rules,
    transposed into the user's natural vocal register (`semitone_shift`).
    """
    vp = voice_profile or {}
    semitone_shift = int(vp.get("semitone_shift", 0))
    base_note_hz = float(vp.get("base_note_hz", TARGET_NOTE_HZ))

    sample_lookup = {s["id"]: s for s in samples_meta}
    lead_in_sec = 0.65
    total_beats = 49.0
    total_dur_sec = lead_in_sec + total_beats * BEAT_SEC + 1.0
    total_samples = int(total_dur_sec * SAMPLE_RATE)

    vocal_bus = np.zeros(total_samples, dtype=np.float64)
    accompaniment_bus = np.zeros(total_samples, dtype=np.float64)
    timed_lyrics: List[Dict[str, Any]] = []

    analysis_cache: Dict[str, Dict[str, Any]] = {}

    for idx, note in enumerate(DAISY_BELL_NOTES):
        sid = note["sample_id"]
        meta = sample_lookup.get(sid) or samples_meta[0]
        wav_path = voicebank_dir / meta["filename"]
        oto = meta["oto"]
        ptype = meta.get("type", "cv")

        raw_wav, _sr = sf.read(str(wav_path), dtype="float64", always_2d=True)
        mono = np.mean(raw_wav, axis=1)

        note_start_sec = lead_in_sec + note["start_beat"] * BEAT_SEC
        raw_note_dur_sec = note["dur_beats"] * BEAT_SEC

        # Authentic UTAU wavtool boundary trimming so adjacent notes crossfade cleanly
        effective_vowel_sec = raw_note_dur_sec
        if idx + 1 < len(DAISY_BELL_NOTES):
            next_note = DAISY_BELL_NOTES[idx + 1]
            next_start_sec = lead_in_sec + next_note["start_beat"] * BEAT_SEC
            gap_to_next = next_start_sec - (note_start_sec + raw_note_dur_sec)
            if abs(gap_to_next) < 0.05:
                next_meta = sample_lookup.get(next_note["sample_id"]) or samples_meta[0]
                next_preutt_sec = float(next_meta["oto"].get("preutterance", 55.0)) / 1000.0
                next_ovl_sec = float(next_meta["oto"].get("overlap", 22.0)) / 1000.0
                trim_pullback = max(0.0, next_preutt_sec - next_ovl_sec * 1.05)
                effective_vowel_sec = max(0.22, raw_note_dur_sec - trim_pullback)

        user_midi = int(note["midi"]) + semitone_shift
        target_hz = midi_to_hz(user_midi)
        prev_midi = int(DAISY_BELL_NOTES[idx - 1]["midi"]) + semitone_shift if idx > 0 else user_midi
        prev_hz = midi_to_hz(prev_midi)

        slur_hz: Optional[float] = None
        slur_sec: Optional[float] = None
        if "slur_to_midi" in note:
            slur_hz = midi_to_hz(int(note["slur_to_midi"]) + semitone_shift)
            slur_sec = float(note.get("slur_at_beat_offset", 3.0)) * BEAT_SEC

        rendered_note, preutt_sec = _resample_utau_note(
            mono,
            sample_id=sid,
            oto=oto,
            ptype=ptype,
            base_note_hz=base_note_hz,
            target_hz=target_hz,
            prev_hz=prev_hz,
            target_vowel_sec=effective_vowel_sec,
            slur_hz=slur_hz,
            slur_sec=slur_sec,
            analysis_cache=analysis_cache,
            sr=SAMPLE_RATE,
        )

        place_start = max(0, int((note_start_sec - preutt_sec) * SAMPLE_RATE))
        place_end = min(total_samples, place_start + len(rendered_note))
        seg_len = place_end - place_start
        if seg_len > 0:
            vocal_bus[place_start:place_end] += rendered_note[:seg_len]

        timed_lyrics.append({
            "index": idx,
            "lyric": note["lyric"],
            "alias": note["alias"],
            "sample_id": sid,
            "midi": user_midi,
            "freq_hz": round(target_hz, 1),
            "start_sec": round(note_start_sec, 3),
            "end_sec": round(note_start_sec + raw_note_dur_sec, 3),
        })

    _render_waltz_accompaniment(accompaniment_bus, lead_in_sec, semitone_shift, SAMPLE_RATE)

    # Apply gentle room reverb and soft-knee limiting so there is ZERO clipping on high notes
    vocal_bus = _apply_studio_reverb(vocal_bus, SAMPLE_RATE, wet=0.11)
    vocal_bus = soft_limit_audio(vocal_bus, target_peak=0.74)

    mixed = vocal_bus + accompaniment_bus * 0.11
    save_utau_wav(output_wav_path, mixed)
    return timed_lyrics


def _resample_utau_note(
    sample_wav: np.ndarray,
    sample_id: str,
    oto: Dict[str, float],
    ptype: str,
    base_note_hz: float,
    target_hz: float,
    prev_hz: float,
    target_vowel_sec: float,
    slur_hz: Optional[float],
    slur_sec: Optional[float],
    analysis_cache: Dict[str, Dict[str, Any]],
    sr: int = SAMPLE_RATE,
) -> tuple[np.ndarray, float]:
    """
    Natural Human-Contour UTAU Resampler:
    - Preserves the user's real spoken consonant attack and vowel transition at 1x speed.
    - Stretches the interior vowel nucleus via smooth forward log-spectral interpolation
      (preserving natural human spectral micro-life without 1-frame robotic buzz or "wayo" loops).
    - Tames high-note formant resonance spikes and applies equal-loudness scaling so high notes
      never clip or distort.
    """
    total_ms = (len(sample_wav) / float(sr)) * 1000.0
    offset_ms = float(oto.get("offset", 65.0))
    cutoff_ms = float(oto.get("cutoff", -90.0))
    preutt_ms = float(oto.get("preutterance", 55.0))
    overlap_ms = float(oto.get("overlap", 22.0))
    consonant_ms = float(oto.get("consonant", 145.0))

    end_ms = total_ms + cutoff_ms if cutoff_ms < 0 else total_ms - cutoff_ms
    s_idx = max(0, int((offset_ms / 1000.0) * sr))
    e_idx = min(len(sample_wav), max(s_idx + int(0.1 * sr), int((end_ms / 1000.0) * sr)))
    active = sample_wav[s_idx:e_idx].astype(np.float64)

    preutt_sec = preutt_ms / 1000.0
    fixed_sec = max(preutt_sec + 0.05, consonant_ms / 1000.0)

    if HAS_PYWORLD and pw is not None and len(active) > int(0.08 * sr):
        frame_period = 5.0
        if sample_id in analysis_cache:
            cached = analysis_cache[sample_id]
            sp = cached["sp"]
            ap = cached["ap"]
            f0_for_analysis = cached["f0"]
            micro_contour = cached["micro_contour"]
            cons_end_frame = cached["cons_end_frame"]
            n_frames = len(sp)
        else:
            n_frames = int(len(active) / (sr * (frame_period / 1000.0))) + 1
            t = np.arange(n_frames, dtype=np.float64) * (frame_period / 1000.0)
            t = t[t * sr < len(active)]
            n_frames = len(t)

            _f0_raw, _t_raw = pw.dio(active, sr, f0_floor=55.0, f0_ceil=520.0, frame_period=frame_period)
            f0_raw = pw.stonemask(active, _f0_raw, _t_raw, sr)[:n_frames]

            cons_end_frame = 2 if ptype == "vowel" else min(n_frames // 3, int((preutt_ms * 0.85) / frame_period))
            voiced_vals = f0_raw[f0_raw > 58.0]
            med_f0 = float(np.median(voiced_vals)) if len(voiced_vals) > 0 else base_note_hz

            f0_for_analysis = np.full(n_frames, med_f0, dtype=np.float64)
            if cons_end_frame > 0:
                f0_for_analysis[:cons_end_frame] = np.where(f0_raw[:cons_end_frame] > 58.0, med_f0, 0.0)

            # Extract the user's subtle natural vocal-cord micro-intonation
            raw_ratio = np.where(f0_raw > 58.0, f0_raw / med_f0, 1.0)
            micro_contour = np.clip(raw_ratio ** 0.28, 0.965, 1.035)

            sp = np.maximum(pw.cheaptrick(active, f0_for_analysis, t, sr), 1e-10)
            ap = np.clip(pw.d4c(active, f0_for_analysis, t, sr), 0.001, 0.85)
            ap[cons_end_frame:] = np.minimum(ap[cons_end_frame:], 0.22)

            analysis_cache[sample_id] = {
                "sp": sp,
                "ap": ap,
                "f0": f0_for_analysis,
                "micro_contour": micro_contour,
                "cons_end_frame": cons_end_frame,
            }

        preutt_frames = int((preutt_sec * 1000.0) / frame_period)
        target_total_frames = max(preutt_frames + 18, preutt_frames + int((target_vowel_sec * 1000.0) / frame_period))

        coda_frames = min(14, max(4, n_frames // 7))
        max_fixed = max(cons_end_frame + 5, target_total_frames - coda_frames - 6)
        fixed_frames = min(n_frames - coda_frames - 4, max_fixed, int((fixed_sec * 1000.0) / frame_period))
        fixed_frames = max(cons_end_frame + 4, fixed_frames)

        sustain_frames_needed = max(4, target_total_frames - fixed_frames - coda_frames)

        # Smoothly time-interpolate the real interior vowel nucleus (frames fixed_frames .. n_frames - coda_frames)
        # in log-spectral space (forward-only: preserves organic human vowel texture with zero looping!)
        interior_sp = sp[fixed_frames : n_frames - coda_frames]
        interior_ap = ap[fixed_frames : n_frames - coda_frames]
        interior_mc = micro_contour[fixed_frames : n_frames - coda_frames]

        sustain_sp = stretch_log_spectrum_smooth(interior_sp, sustain_frames_needed)
        sustain_ap = stretch_log_spectrum_smooth(interior_ap, sustain_frames_needed)

        mc_src_t = np.linspace(0.0, 1.0, max(1, len(interior_mc)))
        mc_dst_t = np.linspace(0.0, 1.0, sustain_frames_needed)
        sustain_mc = np.interp(mc_dst_t, mc_src_t, interior_mc) if len(interior_mc) > 1 else np.ones(sustain_frames_needed)

        sp_out = np.vstack([sp[:fixed_frames], sustain_sp, sp[-coda_frames:]])
        ap_out = np.vstack([ap[:fixed_frames], sustain_ap, ap[-coda_frames:]])
        mc_out = np.concatenate([micro_contour[:fixed_frames], sustain_mc, micro_contour[-coda_frames:]])

        # High-note anti-clipping spectral taming:
        # When shifting up significantly, compress sharp resonant formant spikes in log-spectrum
        pitch_ratio = target_hz / max(80.0, base_note_hz)
        if pitch_ratio > 1.18:
            log_sp = np.log(np.maximum(sp_out, 1e-10))
            mean_log = np.mean(log_sp, axis=1, keepdims=True)
            # Gently compress spectral peaks above the frame mean so no single harmonic clips
            excess = np.maximum(0.0, log_sp - (mean_log + 2.2))
            tame_factor = float(np.clip((pitch_ratio - 1.15) * 0.38, 0.0, 0.32))
            sp_out = np.exp(log_sp - excess * tame_factor)

        n_out = len(sp_out)
        frame_times = np.arange(n_out, dtype=np.float64) * (frame_period / 1000.0)

        # Build expressive human F0 contour (portamento + user's real micro-contour + gentle vibrato)
        glide = np.full(n_out, target_hz, dtype=np.float64)
        glide_frames = min(n_out, max(1, int(0.050 * 1000.0 / frame_period)))
        glide[:glide_frames] = np.linspace(
            0.38 * prev_hz + 0.62 * target_hz,
            target_hz,
            glide_frames,
        )

        if slur_hz is not None and slur_sec is not None:
            slur_start_f = preutt_frames + int((slur_sec * 1000.0) / frame_period)
            slur_trans_f = int(0.090 * 1000.0 / frame_period)
            if slur_start_f < n_out:
                end_tr = min(n_out, slur_start_f + slur_trans_f)
                glide[slur_start_f:end_tr] = np.linspace(target_hz, slur_hz, end_tr - slur_start_f)
                if end_tr < n_out:
                    glide[end_tr:] = slur_hz

        # Apply the user's real vocal-cord micro-intonation
        glide = glide * mc_out

        if target_vowel_sec >= 0.70:
            vib_env = np.clip((frame_times - 0.32) / 0.25, 0.0, 1.0)
            vib_mod = 1.0 + 0.0065 * vib_env * np.sin(2.0 * np.pi * 5.2 * frame_times)
            glide = glide * vib_mod

        voiced_out = np.ones(n_out, dtype=bool)
        if cons_end_frame > 0:
            voiced_out[:cons_end_frame] = f0_for_analysis[:cons_end_frame] > 0.0
        f0_out = np.where(voiced_out, glide, 0.0)

        rendered = pw.synthesize(
            np.ascontiguousarray(f0_out, dtype=np.float64),
            np.ascontiguousarray(np.maximum(sp_out, 1e-10), dtype=np.float64),
            np.ascontiguousarray(np.clip(ap_out, 0.001, 0.85), dtype=np.float64),
            sr,
            frame_period=frame_period,
        )
        rendered = np.nan_to_num(rendered, nan=0.0, posinf=0.0, neginf=0.0)

        # Splice the user's raw un-vocoded consonant attack at the start of CV notes for natural consonant clarity
        hop_samples = int((frame_period / 1000.0) * sr)
        cons_samples = cons_end_frame * hop_samples
        if ptype != "vowel" and cons_samples > 32 and len(active) > cons_samples + 32 and len(rendered) > cons_samples + 32:
            xfade = min(cons_samples // 2, int(0.014 * sr))
            raw_cons = active[:cons_samples]
            fade_o = np.linspace(1.0, 0.0, xfade)
            fade_i = np.linspace(0.0, 1.0, xfade)
            rendered[: cons_samples - xfade] = raw_cons[: cons_samples - xfade]
            rendered[cons_samples - xfade : cons_samples] = (
                raw_cons[cons_samples - xfade : cons_samples] * fade_o
                + rendered[cons_samples - xfade : cons_samples] * fade_i
            )
    else:
        pitch_ratio = target_hz / base_note_hz
        shifted = signal.resample(active, max(64, int(len(active) / pitch_ratio)))
        target_len = int((preutt_sec + target_vowel_sec) * sr)
        rendered = signal.resample(shifted, max(64, target_len))

    fade_in = max(32, int((max(16.0, overlap_ms) / 1000.0) * sr))
    fade_out = max(64, int(0.048 * sr))
    if len(rendered) > fade_in + fade_out:
        rendered[:fade_in] *= np.linspace(0.0, 1.0, fade_in)
        rendered[-fade_out:] *= np.linspace(1.0, 0.0, fade_out)

    # Equal-loudness high-note compensation so higher notes never jump out or clip
    pitch_gain = float(np.clip((base_note_hz / max(80.0, target_hz)) ** 0.26, 0.72, 1.05))
    note_peak = 0.68 * pitch_gain
    return soft_limit_audio(rendered, target_peak=note_peak), preutt_sec


def _render_waltz_accompaniment(
    bus: np.ndarray,
    lead_in_sec: float,
    semitone_shift: int = 0,
    sr: int = SAMPLE_RATE,
) -> None:
    chords = [
        (48, [60, 64, 67]),
        (43, [60, 64, 67]),
        (48, [60, 64, 67]),
        (43, [60, 64, 67]),
        (53, [60, 65, 69]),
        (43, [59, 62, 65]),
        (48, [60, 64, 67]),
        (48, [60, 64, 67]),
        (53, [60, 65, 69]),
        (50, [62, 65, 69]),
        (43, [59, 62, 67]),
        (48, [60, 64, 67]),
        (45, [61, 64, 69]),
        (50, [60, 66, 69]),
        (43, [59, 62, 67]),
        (43, [59, 62, 67]),
    ]

    # Match the exact key (pitch class) of the vocal semitone_shift while keeping chords in middle octave
    acc_shift = semitone_shift
    while acc_shift < -6:
        acc_shift += 12
    for bar_idx, (bass_midi, triad_midis) in enumerate(chords):
        bar_beat = bar_idx * 3.0
        b_m = bass_midi + acc_shift
        t_m = [m + acc_shift for m in triad_midis]
        _add_bell_tone(bus, lead_in_sec + bar_beat * BEAT_SEC, BEAT_SEC * 0.9, [b_m], sr, amp=0.42)
        _add_bell_tone(bus, lead_in_sec + (bar_beat + 1.0) * BEAT_SEC, BEAT_SEC * 0.75, t_m, sr, amp=0.22)
        _add_bell_tone(bus, lead_in_sec + (bar_beat + 2.0) * BEAT_SEC, BEAT_SEC * 0.75, t_m, sr, amp=0.18)


def _add_bell_tone(
    bus: np.ndarray,
    start_sec: float,
    dur_sec: float,
    midis: List[int],
    sr: int,
    amp: float = 0.35,
) -> None:
    s_idx = int(start_sec * sr)
    n = int(dur_sec * sr)
    if s_idx >= len(bus) or n <= 0:
        return
    e_idx = min(len(bus), s_idx + n)
    t = np.arange(e_idx - s_idx, dtype=np.float64) / sr
    env = np.exp(-3.8 * t / max(0.15, dur_sec))
    attack = min(len(t), int(0.01 * sr))
    env[:attack] *= np.linspace(0.0, 1.0, attack)

    tone = np.zeros_like(t)
    for m in midis:
        f = midi_to_hz(m)
        tone += np.sin(2.0 * np.pi * f * t) + 0.25 * np.sin(2.0 * np.pi * (2.0 * f) * t)
    bus[s_idx:e_idx] += (tone / max(1, len(midis))) * env * amp


def _apply_studio_reverb(audio: np.ndarray, sr: int = SAMPLE_RATE, wet: float = 0.11) -> np.ndarray:
    delays_ms = [29.0, 43.0, 61.0, 89.0]
    gains = [0.45, 0.32, 0.22, 0.14]
    reverb = np.zeros_like(audio)
    for d_ms, g in zip(delays_ms, gains):
        d_samples = int((d_ms / 1000.0) * sr)
        if d_samples < len(audio):
            reverb[d_samples:] += audio[:-d_samples] * g
    return (1.0 - wet) * audio + wet * reverb


def generate_openutau_ustx(character_name: str, singer_id: str, semitone_shift: int = 0) -> str:
    ticks_per_beat = 480
    notes_list = []
    for n in DAISY_BELL_NOTES:
        pos_ticks = int(round(n["start_beat"] * ticks_per_beat))
        dur_ticks = int(round(n["dur_beats"] * ticks_per_beat))
        notes_list.append({
            "position": pos_ticks,
            "duration": dur_ticks,
            "tone": int(n["midi"]) + semitone_shift,
            "lyric": n["alias"],
            "pitch": {
                "data": [{"x": -40, "y": 0, "shape": "io"}, {"x": 40, "y": 0, "shape": "io"}],
                "snap_first": True,
            },
            "vibrato": {
                "length": 65 if n["dur_beats"] >= 2.0 else 0,
                "period": 175,
                "depth": 22,
                "in": 15,
                "out": 15,
                "shift": 0,
                "drift": 0,
            },
        })

    ustx_doc = {
        "name": f"Daisy Bell - {character_name}",
        "comment": "Generated by OpenUTAU Voicebank Studio (GreatUniHack)",
        "output_dir": "Vocal",
        "cache_dir": "UCache",
        "ustx_version": "0.6",
        "resolution": ticks_per_beat,
        "bpm": BPM,
        "beat_per_bar": 3,
        "beat_unit": 4,
        "expressions": {
            "dyn": {"name": "dynamics (curve)", "abbr": "dyn", "type": "Curve", "min": -240, "max": 120, "default_value": 0, "is_flag": False},
            "pitd": {"name": "pitch deviation (curve)", "abbr": "pitd", "type": "Curve", "min": -1200, "max": 1200, "default_value": 0, "is_flag": False},
            "clr": {"name": "voice color", "abbr": "clr", "type": "Options", "min": 0, "max": 0, "default_value": 0, "is_flag": False, "options": [""]},
            "eng": {"name": "resampler engine", "abbr": "eng", "type": "Options", "min": 0, "max": 1, "default_value": 0, "is_flag": False, "options": ["", "worldline"]},
            "vel": {"name": "velocity", "abbr": "vel", "type": "Numerical", "min": 0, "max": 200, "default_value": 100, "is_flag": False},
            "vol": {"name": "volume", "abbr": "vol", "type": "Numerical", "min": 0, "max": 200, "default_value": 100, "is_flag": False},
            "atk": {"name": "attack", "abbr": "atk", "type": "Numerical", "min": 0, "max": 200, "default_value": 100, "is_flag": False},
            "dec": {"name": "decay", "abbr": "dec", "type": "Numerical", "min": 0, "max": 100, "default_value": 0, "is_flag": False},
            "gen": {"name": "gender", "abbr": "gen", "type": "Numerical", "min": -100, "max": 100, "default_value": 0, "is_flag": True, "flag": "g"},
            "bre": {"name": "breath", "abbr": "bre", "type": "Numerical", "min": 0, "max": 100, "default_value": 0, "is_flag": True, "flag": "B"},
        },
        "exp_selectors": ["dyn", "pitd", "clr", "eng", "vel"],
        "exp_primary": 0,
        "exp_secondary": 1,
        "key": 0,
        "time_signatures": [{"bar_position": 0, "beat_per_bar": 3, "beat_unit": 4}],
        "tempos": [{"position": 0, "bpm": BPM}],
        "tracks": [
            {
                "singer": character_name,
                "phonemizer": "OpenUtau.Core.DefaultPhonemizer",
                "renderer_settings": {"renderer": "WORLDLINE-R"},
                "track_name": "Daisy Bell Lead",
                "track_color": "Blue",
                "mute": False,
                "solo": False,
                "volume": 0,
            }
        ],
        "voice_parts": [
            {
                "name": "Daisy Bell Chorus",
                "comment": "Uses generated ARPAbet / CV aliases directly",
                "track_no": 0,
                "position": 0,
                "notes": notes_list,
                "curves": [],
            }
        ],
        "wave_parts": [],
    }
    return yaml.safe_dump(ustx_doc, sort_keys=False, allow_unicode=True)


def generate_classic_ust(character_name: str, semitone_shift: int = 0) -> str:
    ticks_per_beat = 480
    lines = [
        "[#VERSION]",
        "UST Version1.2",
        "[#SETTING]",
        f"Tempo={BPM:.2f}",
        "Tracks=1",
        f"ProjectName=Daisy Bell ({character_name})",
        f"VoiceDir=%VOICE%{character_name}",
        "OutFile=Daisy_Bell.wav",
        "CacheDir=Daisy_Bell.cache",
        "Tool1=wavtool.exe",
        "Tool2=resampler.exe",
        "Mode2=True",
    ]

    cur_tick = 0
    note_idx = 0
    for n in DAISY_BELL_NOTES:
        start_tick = int(round(n["start_beat"] * ticks_per_beat))
        dur_tick = int(round(n["dur_beats"] * ticks_per_beat))
        if start_tick > cur_tick:
            rest_ticks = start_tick - cur_tick
            lines.extend([
                f"[#{note_idx:04d}]",
                f"Length={rest_ticks}",
                "Lyric=R",
                "NoteNum=60",
            ])
            note_idx += 1
            cur_tick = start_tick

        lines.extend([
            f"[#{note_idx:04d}]",
            f"Length={dur_tick}",
            f"Lyric={n['alias']}",
            f"NoteNum={int(n['midi']) + semitone_shift}",
            "PreUtterance=",
            "Intensity=100",
            "Modulation=0",
        ])
        note_idx += 1
        cur_tick += dur_tick

    lines.append("[#TRACKEND]")
    return "\r\n".join(lines) + "\r\n"
