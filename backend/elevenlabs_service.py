"""
ElevenLabs Voice Cloning & Direct User-Voice Corpus Phoneme Synthesis Service.
Includes:
- Instant Voice Cloning (POST /v1/voices/add)
- Batched TTS with Character Timestamps (POST /v1/text-to-speech/{voice_id}/with-timestamps)
- Guaranteed temporary voice slot cleanup (DELETE /v1/voices/{voice_id})
- Local Direct User-Voice Syllable & Spectral Corpus Extraction (`UserVoiceCorpus`) so
  Local Mode builds the voicebank directly from the user's real recorded voice frames!
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx
import numpy as np
import soundfile as sf

from .audio_processor import (
    SAMPLE_RATE,
    UserVoiceCorpus,
    compute_oto_parameters,
    flatten_pitch_and_prepare_sample,
    load_audio_mono_44k,
    save_utau_wav,
    slice_batch_with_timestamps,
)
from .phonemes import PHONEME_INVENTORY, get_batches

ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"


class ElevenLabsError(RuntimeError):
    """Raised when ElevenLabs synthesis fails and local fallback is disabled."""


async def generate_voicebank_samples(
    reference_audio: np.ndarray,
    voice_profile: Dict[str, Any],
    output_dir: Path,
    character_name: str,
    api_key: Optional[str] = None,
    use_demo_mode: bool = False,
    auto_cleanup_voice: bool = True,
    allow_fallback: bool = True,
    progress_cb: Optional[Callable[[str, int], None]] = None,
) -> Dict[str, Any]:
    """
    Generate all 48 UTAU phoneme WAV files and compute their oto.ini parameters.
    Returns metadata including sample list, oto entries, and synthesis mode used.
    If allow_fallback is False, ElevenLabs failures raise ElevenLabsError instead of
    silently switching to local mode.
    """
    resolved_key = (api_key or os.getenv("ELEVENLABS_API_KEY") or "").strip()
    if not use_demo_mode and not resolved_key and not allow_fallback:
        raise ElevenLabsError("No ElevenLabs API key provided.")
    base_note_hz = float(voice_profile.get("base_note_hz", 174.61))
    base_note_name = str(voice_profile.get("base_note_name", "F3"))

    mode_used = "local_user_voice_corpus"
    mode_detail = (
        f"Vosk Word-Aligned + Praat TD-PSOLA Extraction (Base Pitch: {base_note_name} / {base_note_hz} Hz)"
    )
    cloned_voice_id: Optional[str] = None
    voice_cleaned_up = False

    raw_clips_by_id: Dict[str, np.ndarray] = {}
    unshifted_word_clips: Dict[str, np.ndarray] = {}
    from_local_corpus: set[str] = set()

    if not use_demo_mode and resolved_key:
        try:
            if progress_cb:
                progress_cb("Cloning voice via ElevenLabs Instant Voice Cloning...", 18)

            ref_buf = io.BytesIO()
            sf.write(ref_buf, reference_audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
            ref_buf.seek(0)

            headers = {"xi-api-key": resolved_key}
            async with httpx.AsyncClient(timeout=90.0) as client:
                files = [("files", ("reference.wav", ref_buf.read(), "audio/wav"))]
                data = {
                    "name": f"OpenUTAU_{character_name[:24]}",
                    "description": "Temporary OpenUTAU voicebank clone generated at GreatUniHack",
                }
                clone_resp = await client.post(
                    f"{ELEVENLABS_BASE_URL}/voices/add",
                    headers=headers,
                    data=data,
                    files=files,
                )
                if clone_resp.status_code != 200:
                    raise RuntimeError(
                        f"ElevenLabs /voices/add returned {clone_resp.status_code}: {clone_resp.text[:240]}"
                    )
                cloned_voice_id = clone_resp.json().get("voice_id")
                if not cloned_voice_id:
                    raise RuntimeError("ElevenLabs did not return a voice_id")

                batches = get_batches()
                total_batches = len(batches)
                for b_idx, (batch_num, items) in enumerate(sorted(batches.items()), start=1):
                    if progress_cb:
                        pct = 25 + int((b_idx / total_batches) * 35)
                        progress_cb(
                            f"Synthesizing ARPAbet carrier batch {b_idx}/{total_batches} via ElevenLabs...",
                            pct,
                        )

                    prompt_text = " . ".join(item["word"] for item in items) + " ."
                    tts_payload = {
                        "text": prompt_text,
                        "model_id": "eleven_multilingual_v2",
                        "voice_settings": {
                            "stability": 0.86,
                            "similarity_boost": 0.88,
                            "style": 0.0,
                            "use_speaker_boost": True,
                        },
                    }
                    tts_url = f"{ELEVENLABS_BASE_URL}/text-to-speech/{cloned_voice_id}/with-timestamps"
                    tts_resp = await client.post(tts_url, headers=headers, json=tts_payload)

                    if tts_resp.status_code != 200:
                        raise RuntimeError(
                            f"ElevenLabs TTS batch {batch_num} failed ({tts_resp.status_code}): {tts_resp.text[:200]}"
                        )

                    body = tts_resp.json()
                    audio_b64 = body.get("audio_base64", "")
                    alignment = body.get("alignment") or body.get("normalized_alignment")
                    audio_bytes = base64.b64decode(audio_b64)
                    batch_audio = load_audio_mono_44k(audio_bytes, filename_hint="batch.mp3")

                    sliced_clips = slice_batch_with_timestamps(batch_audio, alignment, items, SAMPLE_RATE)
                    for item, clip in zip(items, sliced_clips):
                        raw_clips_by_id[item["id"]] = clip
                        unshifted_word_clips[item["id"]] = clip

                mode_used = "elevenlabs_ivc"
                mode_detail = (
                    f"ElevenLabs Instant Voice Cloning + Praat TD-PSOLA ({base_note_name} / {base_note_hz} Hz)"
                )

        except Exception as exc:
            if not allow_fallback:
                raise ElevenLabsError(str(exc)[:300]) from exc
            mode_used = "local_user_voice_corpus"
            mode_detail = (
                f"Vosk Word-Aligned + Praat TD-PSOLA ({base_note_name} / {base_note_hz} Hz — "
                f"ElevenLabs fallback: {str(exc)[:110]})"
            )
            raw_clips_by_id.clear()
            unshifted_word_clips.clear()

        finally:
            if cloned_voice_id and auto_cleanup_voice and resolved_key:
                try:
                    async with httpx.AsyncClient(timeout=20.0) as cleanup_client:
                        del_resp = await cleanup_client.delete(
                            f"{ELEVENLABS_BASE_URL}/voices/{cloned_voice_id}",
                            headers={"xi-api-key": resolved_key},
                        )
                        voice_cleaned_up = del_resp.status_code == 200
                except Exception:
                    pass

    # Extract directly from the user's recorded voice corpus for Local Mode (or any missing clips)
    if len(raw_clips_by_id) < len(PHONEME_INVENTORY):
        if progress_cb:
            progress_cb("Aligning spoken words & extracting raw glottal waveforms...", 45)
        corpus = UserVoiceCorpus(reference_audio, voice_profile, sr=SAMPLE_RATE)
        for item in PHONEME_INVENTORY:
            pid = item["id"]
            if pid not in raw_clips_by_id:
                unshifted_word_clips[pid] = corpus.get_raw_word_slice(item)
                raw_clips_by_id[pid] = corpus.synthesize_phoneme_from_user_voice(item)
                from_local_corpus.add(pid)

    if progress_cb:
        progress_cb(
            f"Normalizing pitch to {base_note_name} ({base_note_hz} Hz) & computing oto.ini boundaries...",
            68,
        )

    samples_meta: List[Dict[str, Any]] = []
    oto_lines: List[str] = []

    for item in PHONEME_INVENTORY:
        pid = item["id"]
        wav_filename = f"{pid}.wav"
        wav_path = output_dir / wav_filename

        raw_clip = raw_clips_by_id[pid]
        processed_audio, cons_dur_ms = flatten_pitch_and_prepare_sample(
            raw_clip,
            item,
            target_f0=base_note_hz,
            sr=SAMPLE_RATE,
            already_flattened=(pid in from_local_corpus),
        )
        save_utau_wav(wav_path, processed_audio)

        oto_params = compute_oto_parameters(processed_audio, item, cons_dur_ms)

        for alias in item["aliases"]:
            line = (
                f"{wav_filename}={alias},"
                f"{oto_params['offset']},"
                f"{oto_params['consonant']},"
                f"{oto_params['cutoff']},"
                f"{oto_params['preutterance']},"
                f"{oto_params['overlap']}"
            )
            oto_lines.append(line)

        n_pts = 160
        chunk = max(1, len(processed_audio) // n_pts)
        peaks = [
            round(float(np.max(np.abs(processed_audio[i * chunk : (i + 1) * chunk]))), 4)
            for i in range(min(n_pts, len(processed_audio) // chunk))
        ]

        samples_meta.append({
            "id": pid,
            "filename": wav_filename,
            "word": item["word"],
            "arpabet": item["arpabet"],
            "type": item["type"],
            "aliases": item["aliases"],
            "description": item["description"],
            "oto": oto_params,
            "peaks": peaks,
        })

    return {
        "mode_used": mode_used,
        "mode_detail": mode_detail,
        "voice_cleaned_up": voice_cleaned_up,
        "samples": samples_meta,
        "oto_lines": oto_lines,
        "raw_word_clips": unshifted_word_clips,
    }
