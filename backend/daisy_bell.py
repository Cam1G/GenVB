"""
UTAU Time-Domain Resampler Engine & Project Exporter for "Daisy Bell (Bicycle Built for Two)" (1892 / IBM 7094 1961).
- Automatically centers the melody in the user's natural speaking/chest register.
- Uses Single-Pass Phase-Locked Time-Domain WSOLA (`wsola_stretch_phase_locked`) + Praat TD-PSOLA
  (`psola_pitch_transfer`) directly on the user's raw Vosk-aligned spoken words—preserving 100% of
  their real glottal waveform phase, regional/British accent, and vocal timbre with ZERO `pyworld`
  spectral vocoder buzz!
- Includes Natural Human Pitch Splines (onset scoops, phrase-end breath releases, retained spoken
  micro-intonation, and organic vocal drift) + Studio Soft-Knee Limiting for zero clipping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.signal as signal
import soundfile as sf
import yaml

from .audio_processor import (
    SAMPLE_RATE,
    TARGET_NOTE_HZ,
    psola_pitch_transfer,
    save_utau_wav,
    soft_limit_audio,
    wsola_stretch_phase_locked,
)

BPM = 152.0  # Lively, natural 3/4 waltz tempo so notes aren't over-stretched
BEAT_SEC = 60.0 / BPM

# "Daisy Bell" Chorus Score (Reference Key: C Major, automatically transposed by user's semitone_shift
# so the center of the melody sits right at the user's natural vocal register).
# Phrase-ending notes have natural breathing durations (2.2 - 3.6 beats) so the voice breathes between phrases!
DAISY_BELL_NOTES: List[Dict[str, Any]] = [
    # Phrase 1: "Dai - sy, Dai - sy,"
    {"lyric": "Dai-", "alias": "d ey", "sample_id": "16_d_ey", "midi": 72, "start_beat": 0.0, "dur_beats": 2.9, "phrase_start": True},
    {"lyric": "-sy,", "alias": "z iy", "sample_id": "17_z_iy", "midi": 67, "start_beat": 3.0, "dur_beats": 2.3, "phrase_end": True},
    {"lyric": "Dai-", "alias": "d ey", "sample_id": "16_d_ey", "midi": 64, "start_beat": 6.0, "dur_beats": 2.9, "phrase_start": True},
    {"lyric": "-sy,", "alias": "z iy", "sample_id": "17_z_iy", "midi": 60, "start_beat": 9.0, "dur_beats": 2.3, "phrase_end": True},

    # Phrase 2: "give me your an - swer do!"
    {"lyric": "give", "alias": "g ih", "sample_id": "18_g_ih", "midi": 62, "start_beat": 12.0, "dur_beats": 1.0, "phrase_start": True},
    {"lyric": "me", "alias": "m iy", "sample_id": "19_m_iy", "midi": 64, "start_beat": 13.0, "dur_beats": 1.0},
    {"lyric": "your", "alias": "y ao", "sample_id": "20_y_ao", "midi": 65, "start_beat": 14.0, "dur_beats": 1.0},
    {"lyric": "an-", "alias": "- ae n", "sample_id": "21_ae_n", "midi": 62, "start_beat": 15.0, "dur_beats": 2.0},
    {"lyric": "-swer", "alias": "s er", "sample_id": "22_s_er", "midi": 65, "start_beat": 17.0, "dur_beats": 1.0},
    {"lyric": "do!", "alias": "d uw", "sample_id": "23_d_uw", "midi": 60, "start_beat": 18.0, "dur_beats": 3.3, "phrase_end": True},

    # Phrase 3: "I'm half cra - zy,"
    {"lyric": "I'm", "alias": "ay m", "sample_id": "06_ay", "midi": 69, "start_beat": 24.0, "dur_beats": 2.8, "phrase_start": True},
    {"lyric": "half", "alias": "hh ae", "sample_id": "24_hh_ae", "midi": 74, "start_beat": 27.0, "dur_beats": 2.9},
    {"lyric": "cra-", "alias": "k r", "sample_id": "25_k_r_ey", "midi": 71, "start_beat": 30.0, "dur_beats": 2.9},
    {"lyric": "-zy,", "alias": "z iy", "sample_id": "17_z_iy", "midi": 67, "start_beat": 33.0, "dur_beats": 2.3, "phrase_end": True},

    # Phrase 4: "all for the love of you!"
    {"lyric": "all", "alias": "ao l", "sample_id": "48_ao_l", "midi": 64, "start_beat": 36.0, "dur_beats": 1.0, "phrase_start": True},
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
        "slur_at_beat_offset": 2.1,
        "start_beat": 42.0,
        "dur_beats": 3.8,
        "phrase_end": True,
    },
]


def midi_to_hz(midi_note: float) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))


def render_daisy_bell_preview(
    voicebank_dir: Path,
    samples_meta: List[Dict[str, Any]],
    output_wav_path: Path,
    voice_profile: Optional[Dict[str, Any]] = None,
    raw_word_clips: Optional[Dict[str, np.ndarray]] = None,
) -> List[Dict[str, Any]]:
    """
    Synthesize "Daisy Bell" using Single-Pass Phase-Locked Time-Domain WSOLA + Praat TD-PSOLA,
    transposed into the user's natural speaking/chest register (`semitone_shift`).
    """
    vp = voice_profile or {}
    semitone_shift = int(vp.get("semitone_shift", 0))
    base_note_hz = float(vp.get("base_note_hz", TARGET_NOTE_HZ))

    sample_lookup = {s["id"]: s for s in samples_meta}
    lead_in_sec = 0.60
    total_beats = 48.0
    total_dur_sec = lead_in_sec + total_beats * BEAT_SEC + 0.9
    total_samples = int(total_dur_sec * SAMPLE_RATE)

    vocal_bus = np.zeros(total_samples, dtype=np.float64)
    accompaniment_bus = np.zeros(total_samples, dtype=np.float64)
    timed_lyrics: List[Dict[str, Any]] = []

    for idx, note in enumerate(DAISY_BELL_NOTES):
        sid = note["sample_id"]
        meta = sample_lookup.get(sid) or samples_meta[0]
        oto = meta["oto"]
        ptype = meta.get("type", "cv")

        # Prefer the 100% un-shifted raw microphone word slice if available so TD-PSOLA runs only ONCE!
        if raw_word_clips and sid in raw_word_clips and len(raw_word_clips[sid]) > int(0.075 * SAMPLE_RATE):
            active_source = raw_word_clips[sid].astype(np.float64)
            is_raw_slice = True
        else:
            wav_path = voicebank_dir / meta["filename"]
            raw_wav, _sr = sf.read(str(wav_path), dtype="float64", always_2d=True)
            active_source = np.mean(raw_wav, axis=1)
            is_raw_slice = False

        note_start_sec = lead_in_sec + note["start_beat"] * BEAT_SEC
        raw_note_dur_sec = note["dur_beats"] * BEAT_SEC

        # Authentic UTAU wavtool boundary trimming so adjacent connected notes crossfade cleanly
        effective_vowel_sec = raw_note_dur_sec
        if idx + 1 < len(DAISY_BELL_NOTES):
            next_note = DAISY_BELL_NOTES[idx + 1]
            next_start_sec = lead_in_sec + next_note["start_beat"] * BEAT_SEC
            gap_to_next = next_start_sec - (note_start_sec + raw_note_dur_sec)
            if abs(gap_to_next) < 0.05:
                next_meta = sample_lookup.get(next_note["sample_id"]) or samples_meta[0]
                next_preutt_sec = float(next_meta["oto"].get("preutterance", 55.0)) / 1000.0
                next_ovl_sec = float(next_meta["oto"].get("overlap", 22.0)) / 1000.0
                trim_pullback = max(0.0, next_preutt_sec - next_ovl_sec * 1.15)
                effective_vowel_sec = max(0.20, raw_note_dur_sec - trim_pullback)

        user_midi = int(note["midi"]) + semitone_shift
        target_hz = midi_to_hz(user_midi)
        prev_midi = int(DAISY_BELL_NOTES[idx - 1]["midi"]) + semitone_shift if idx > 0 else user_midi
        prev_hz = midi_to_hz(prev_midi)

        slur_hz: Optional[float] = None
        slur_sec: Optional[float] = None
        if "slur_to_midi" in note:
            slur_hz = midi_to_hz(int(note["slur_to_midi"]) + semitone_shift)
            slur_sec = float(note.get("slur_at_beat_offset", 2.1)) * BEAT_SEC

        rendered_note, preutt_sec = _resample_utau_note(
            active_source,
            is_raw_slice=is_raw_slice,
            oto=oto,
            ptype=ptype,
            base_note_hz=base_note_hz,
            target_hz=target_hz,
            prev_hz=prev_hz,
            target_vowel_sec=effective_vowel_sec,
            slur_hz=slur_hz,
            slur_sec=slur_sec,
            phrase_start=bool(note.get("phrase_start", False)),
            phrase_end=bool(note.get("phrase_end", False)),
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

    # Apply subtle warm room ambience and linear studio gain staging so there is ZERO clipping
    vocal_bus = _apply_studio_reverb(vocal_bus, SAMPLE_RATE, wet=0.08)
    vocal_bus = soft_limit_audio(vocal_bus, target_peak=0.52, target_rms=0.13)

    mixed = vocal_bus + accompaniment_bus * 0.08
    save_utau_wav(output_wav_path, mixed)
    return timed_lyrics


def _resample_utau_note(
    sample_wav: np.ndarray,
    is_raw_slice: bool,
    oto: Dict[str, float],
    ptype: str,
    base_note_hz: float,
    target_hz: float,
    prev_hz: float,
    target_vowel_sec: float,
    slur_hz: Optional[float],
    slur_sec: Optional[float],
    phrase_start: bool = False,
    phrase_end: bool = False,
    sr: int = SAMPLE_RATE,
) -> tuple[np.ndarray, float]:
    """
    Single-Pass Time-Domain WSOLA + Praat TD-PSOLA Resampler:
    - Expands (or trims) the interior vowel in the time domain via `wsola_stretch_phase_locked`
      so every glottal cycle is copied directly from the speaker's real microphone waveform.
    - Applies a Natural Human Pitch Spline via Praat TD-PSOLA (`psola_pitch_transfer`) with:
      * Smooth cosine-eased onset pitch scoop (or portamento from previous note)
      * 32% retained natural spoken pitch inflection from the user's recording
      * Subtle non-periodic human vocal drift (replacing mechanical sine vibrato)
      * Gentle phrase-end breath pitch relaxation
    """
    preutt_ms = float(oto.get("preutterance", 55.0))
    overlap_ms = float(oto.get("overlap", 22.0))

    if is_raw_slice:
        active = sample_wav.copy()
        preutt_sec = 0.020 if ptype == "vowel" else min(0.065, (len(active) / sr) * 0.22)
    else:
        total_ms = (len(sample_wav) / float(sr)) * 1000.0
        offset_ms = float(oto.get("offset", 65.0))
        cutoff_ms = float(oto.get("cutoff", -90.0))
        end_ms = total_ms + cutoff_ms if cutoff_ms < 0 else total_ms - cutoff_ms
        s_idx = max(0, int((offset_ms / 1000.0) * sr))
        e_idx = min(len(sample_wav), max(s_idx + int(0.08 * sr), int((end_ms / 1000.0) * sr)))
        active = sample_wav[s_idx:e_idx].astype(np.float64)
        preutt_sec = preutt_ms / 1000.0

    target_total_sec = max(preutt_sec + 0.14, preutt_sec + target_vowel_sec)
    target_total_samples = int(target_total_sec * sr)
    min_head_samples = max(int(0.020 * sr), int(preutt_sec * sr))

    # 1. Time-domain phase-locked WSOLA expansion (zero spectral vocoding!)
    time_stretched = wsola_stretch_phase_locked(
        active,
        target_samples=target_total_samples,
        sr=sr,
        min_head_samples=min_head_samples,
    )

    # 2. Natural Human Pitch Spline + Praat TD-PSOLA
    def _human_pitch_spline(times: np.ndarray, orig_f0: np.ndarray, med_f0: float) -> np.ndarray:
        n_pts = len(times)
        if n_pts == 0:
            return orig_f0

        # Base note trajectory with cosine-eased onset scoop / portamento
        curve = np.full(n_pts, target_hz, dtype=np.float64)
        start_hz = target_hz * 0.935 if phrase_start else (0.48 * prev_hz + 0.52 * target_hz)
        scoop_dur = 0.075  # 75ms human onset scoop/glide
        for i, t_sec in enumerate(times):
            rel_t = max(0.0, t_sec - preutt_sec * 0.5)
            if rel_t < scoop_dur:
                p = rel_t / scoop_dur
                ease = 0.5 - 0.5 * np.cos(np.pi * p)
                curve[i] = start_hz + (target_hz - start_hz) * ease

        # Smooth slur transition (e.g. on "you!")
        if slur_hz is not None and slur_sec is not None:
            slur_start_t = preutt_sec + slur_sec
            slur_trans_dur = 0.11
            for i, t_sec in enumerate(times):
                if t_sec >= slur_start_t:
                    p = min(1.0, (t_sec - slur_start_t) / slur_trans_dur)
                    ease = 0.5 - 0.5 * np.cos(np.pi * p)
                    curve[i] = target_hz + (slur_hz - target_hz) * ease

        # Phrase-ending gentle breath pitch relaxation over final 85ms
        if phrase_end and len(times) > 2:
            t_max = float(times[-1])
            rel_start = max(0.15, t_max - 0.085)
            for i, t_sec in enumerate(times):
                if t_sec > rel_start:
                    p = min(1.0, (t_sec - rel_start) / max(0.02, t_max - rel_start))
                    curve[i] *= 1.0 - 0.028 * (p ** 1.5)

        # Retain 32% of the speaker's real spoken micro-prosody
        raw_ratio = np.where(orig_f0 > f0_floor, orig_f0 / max(60.0, med_f0), 1.0)
        micro_contour = np.clip(raw_ratio ** 0.32, 0.955, 1.045)
        curve = curve * micro_contour

        # Subtle non-periodic human vocal drift on sustained notes (replacing robotic sine vibrato)
        if target_vowel_sec >= 0.45:
            drift_env = np.clip((times - 0.18) / 0.20, 0.0, 1.0)
            drift = 1.0 + drift_env * (
                0.0032 * np.sin(2.0 * np.pi * 4.2 * times)
                + 0.0022 * np.sin(2.0 * np.pi * 6.1 * times + 1.1)
            )
            curve = curve * drift

        return curve

    f0_floor = max(55.0, base_note_hz * 0.60)
    f0_ceil = min(380.0, base_note_hz * 1.65)
    rendered = psola_pitch_transfer(time_stretched, sr, _human_pitch_spline, f0_floor=f0_floor, f0_ceil=f0_ceil)

    # 3. Subtle warmth filter on higher notes so upward TD-PSOLA never sounds thin or edgy
    pitch_ratio = target_hz / max(80.0, base_note_hz)
    if pitch_ratio > 1.15 and len(rendered) > 64:
        b_lp, a_lp = signal.butter(1, 3800.0 / (sr / 2.0), btype="low")
        warm = signal.lfilter(b_lp, a_lp, rendered)
        blend = float(np.clip((pitch_ratio - 1.15) * 0.45, 0.0, 0.30))
        rendered = (1.0 - blend) * rendered + blend * warm

    fade_in = max(32, int((max(18.0, overlap_ms) / 1000.0) * sr))
    fade_out_sec = 0.075 if phrase_end else 0.048
    fade_out = max(64, int(fade_out_sec * sr))
    if len(rendered) > fade_in + fade_out:
        t_fi = np.linspace(0.0, np.pi, fade_in)
        rendered[:fade_in] *= 0.5 * (1.0 - np.cos(t_fi))
        t_fo = np.linspace(0.0, np.pi, fade_out)
        rendered[-fade_out:] *= 0.5 * (1.0 + np.cos(t_fo))

    # Equal-loudness high-note compensation + pure linear studio RMS normalization (zero tanh distortion!)
    pitch_gain = float(np.clip((base_note_hz / max(80.0, target_hz)) ** 0.30, 0.68, 1.05))
    note_peak = 0.46 * pitch_gain
    note_rms = 0.115 * pitch_gain
    return soft_limit_audio(rendered, target_peak=note_peak, target_rms=note_rms), preutt_sec


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

    acc_shift = semitone_shift
    while acc_shift < -6:
        acc_shift += 12
    for bar_idx, (bass_midi, triad_midis) in enumerate(chords):
        bar_beat = bar_idx * 3.0
        b_m = bass_midi + acc_shift
        t_m = [m + acc_shift for m in triad_midis]
        _add_bell_tone(bus, lead_in_sec + bar_beat * BEAT_SEC, BEAT_SEC * 0.9, [b_m], sr, amp=0.40)
        _add_bell_tone(bus, lead_in_sec + (bar_beat + 1.0) * BEAT_SEC, BEAT_SEC * 0.75, t_m, sr, amp=0.20)
        _add_bell_tone(bus, lead_in_sec + (bar_beat + 2.0) * BEAT_SEC, BEAT_SEC * 0.75, t_m, sr, amp=0.16)


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


def _apply_studio_reverb(audio: np.ndarray, sr: int = SAMPLE_RATE, wet: float = 0.09) -> np.ndarray:
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
                "length": 55 if n["dur_beats"] >= 2.0 else 0,
                "period": 185,
                "depth": 18,
                "in": 20,
                "out": 20,
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
