"""
FastAPI Backend Server for OpenUTAU Voicebank Studio.
Handles voice recording upload, ElevenLabs voice cloning / local DSP timbre transfer,
WORLD vocoder pitch normalization, oto.ini calculation, "Daisy Bell" vocal synthesis,
OpenUTAU .zip + .ustx packaging, and serves the Vanilla JS Studio Frontend.
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .audio_processor import (
    HAS_PYWORLD,
    SAMPLE_RATE,
    analyze_user_voice,
    load_audio_mono_44k,
    save_utau_wav,
)
from .daisy_bell import render_daisy_bell_preview
from .elevenlabs_service import generate_voicebank_samples
from .packager import package_openutau_voicebank, sanitize_folder_name
from .phonemes import PHONEME_INVENTORY, TELEPROMPTER_SENTENCES

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
SESSIONS_DIR = ROOT_DIR / "generated_sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="OpenUTAU Voicebank Studio",
    description="Turn spoken sentences into an English ARPAbet OpenUTAU voicebank + Daisy Bell performance.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/config")
async def get_studio_config() -> Dict[str, Any]:
    """Return teleprompter sentences, phoneme count, and backend capability flags."""
    return {
        "sentences": TELEPROMPTER_SENTENCES,
        "phoneme_count": len(PHONEME_INVENTORY),
        "has_env_elevenlabs_key": bool((os.getenv("ELEVENLABS_API_KEY") or "").strip()),
        "has_pyworld": HAS_PYWORLD,
        "target_note": "C4 (261.63 Hz)",
        "sample_rate": SAMPLE_RATE,
    }


@app.post("/api/voicebank/generate")
async def create_voicebank(
    audio: UploadFile = File(...),
    character_name: str = Form("Aria_UTAU"),
    author_name: str = Form("GreatUniHack Creator"),
    api_key: Optional[str] = Form(None),
    use_demo_mode: bool = Form(False),
    auto_cleanup_voice: bool = Form(True),
    avatar: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    """
    Full pipeline:
    1. Decode & normalize user voice recording (44.1kHz Mono PCM)
    2. Analyze user vocal profile (median pitch, vocal tract formant scale, harmonics)
    3. Clone voice via ElevenLabs (or local formant transfer) & synthesize 48 ARPAbet phoneme samples
    4. Flatten pitch to C4 (261.63 Hz) via WORLD vocoder & compute oto.ini timing boundaries
    5. Render "Daisy Bell (Bicycle Built for Two)" using the generated UTAU voicebank
    6. Package OpenUTAU .zip (character.txt, character.yaml, oto.ini, avatar.bmp, wavs) + Daisy_Bell.ustx
    """
    raw_bytes = await audio.read()
    if not raw_bytes or len(raw_bytes) < 256:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty or too short.")

    avatar_bytes: Optional[bytes] = None
    if avatar is not None:
        avatar_bytes = await avatar.read()
        if not avatar_bytes:
            avatar_bytes = None

    session_id = uuid.uuid4().hex[:12]
    session_dir = SESSIONS_DIR / session_id
    safe_folder = sanitize_folder_name(character_name)
    voicebank_dir = session_dir / safe_folder
    voicebank_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Decode reference audio
        ref_audio = load_audio_mono_44k(raw_bytes, filename_hint=audio.filename or "recording.webm")
        ref_wav_path = session_dir / "reference.wav"
        save_utau_wav(ref_wav_path, ref_audio)

        # 2. Analyze user vocal characteristics
        voice_profile = analyze_user_voice(ref_audio, SAMPLE_RATE)

        # 3. Synthesize phoneme bank + WORLD pitch flatten + oto.ini computation
        gen_result = await generate_voicebank_samples(
            reference_audio=ref_audio,
            voice_profile=voice_profile,
            output_dir=voicebank_dir,
            character_name=character_name,
            api_key=api_key,
            use_demo_mode=use_demo_mode,
            auto_cleanup_voice=auto_cleanup_voice,
        )

        # 4. Render "Daisy Bell" preview WAV using the generated voicebank + oto.ini
        daisy_wav_path = session_dir / "daisy_bell_preview.wav"
        timed_lyrics = render_daisy_bell_preview(
            voicebank_dir=voicebank_dir,
            samples_meta=gen_result["samples"],
            output_wav_path=daisy_wav_path,
            voice_profile=voice_profile,
        )

        # 5. Package OpenUTAU .zip, character.txt, character.yaml, oto.ini, Daisy_Bell.ustx & .ust
        pack_info = package_openutau_voicebank(
            session_dir=session_dir,
            voicebank_dir=voicebank_dir,
            character_name=character_name,
            author_name=author_name,
            oto_lines=gen_result["oto_lines"],
            voice_profile=voice_profile,
            mode_detail=gen_result["mode_detail"],
            avatar_bytes=avatar_bytes,
        )

        # Attach sample audio URLs for the frontend oto.ini inspector
        samples_with_urls = []
        for s in gen_result["samples"]:
            s_copy = dict(s)
            s_copy["audio_url"] = f"/api/sessions/{session_id}/samples/{s['filename']}"
            samples_with_urls.append(s_copy)

        return {
            "session_id": session_id,
            "character_name": character_name,
            "safe_folder": safe_folder,
            "author_name": author_name,
            "voice_profile": voice_profile,
            "mode_used": gen_result["mode_used"],
            "mode_detail": gen_result["mode_detail"],
            "voice_cleaned_up": gen_result["voice_cleaned_up"],
            "sample_count": len(samples_with_urls),
            "oto_alias_count": len(gen_result["oto_lines"]),
            "urls": {
                "daisy_bell_wav": f"/api/sessions/{session_id}/daisy_bell.wav",
                "voicebank_zip": f"/api/sessions/{session_id}/download/{pack_info['zip_filename']}",
                "daisy_bell_ustx": f"/api/sessions/{session_id}/download/Daisy_Bell.ustx",
                "daisy_bell_ust": f"/api/sessions/{session_id}/download/Daisy_Bell.ust",
                "oto_ini": f"/api/sessions/{session_id}/download/oto.ini",
                "avatar_bmp": f"/api/sessions/{session_id}/avatar.bmp",
                "reference_wav": f"/api/sessions/{session_id}/reference.wav",
            },
            "daisy_bell_lyrics": timed_lyrics,
            "samples": samples_with_urls,
        }

    except Exception as exc:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Voicebank generation failed: {str(exc)}") from exc


@app.get("/api/sessions/{session_id}/daisy_bell.wav")
async def get_daisy_bell_wav(session_id: str) -> FileResponse:
    path = SESSIONS_DIR / session_id / "daisy_bell_preview.wav"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Daisy Bell preview not found")
    return FileResponse(path, media_type="audio/wav", filename="Daisy_Bell_Preview.wav")


@app.get("/api/sessions/{session_id}/reference.wav")
async def get_reference_wav(session_id: str) -> FileResponse:
    path = SESSIONS_DIR / session_id / "reference.wav"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Reference recording not found")
    return FileResponse(path, media_type="audio/wav", filename="Reference_Voice.wav")


@app.get("/api/sessions/{session_id}/avatar.bmp")
async def get_avatar_bmp(session_id: str) -> FileResponse:
    session_dir = SESSIONS_DIR / session_id
    for sub in session_dir.iterdir() if session_dir.is_dir() else []:
        if sub.is_dir() and (sub / "avatar.bmp").is_file():
            return FileResponse(sub / "avatar.bmp", media_type="image/bmp")
    raise HTTPException(status_code=404, detail="Avatar not found")


@app.get("/api/sessions/{session_id}/samples/{filename}")
async def get_sample_wav(session_id: str, filename: str) -> FileResponse:
    session_dir = SESSIONS_DIR / session_id
    if ".." in filename or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    for sub in session_dir.iterdir() if session_dir.is_dir() else []:
        if sub.is_dir() and (sub / filename).is_file():
            return FileResponse(sub / filename, media_type="audio/wav", filename=filename)
    raise HTTPException(status_code=404, detail="Phoneme sample not found")


@app.get("/api/sessions/{session_id}/download/{filename}")
async def download_session_artifact(session_id: str, filename: str) -> FileResponse:
    session_dir = SESSIONS_DIR / session_id
    if ".." in filename or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    direct = session_dir / filename
    if direct.is_file():
        media = "application/zip" if filename.endswith(".zip") else "application/octet-stream"
        return FileResponse(direct, media_type=media, filename=filename)

    for sub in session_dir.iterdir() if session_dir.is_dir() else []:
        if sub.is_dir() and (sub / filename).is_file():
            return FileResponse(sub / filename, media_type="text/plain", filename=filename)

    raise HTTPException(status_code=404, detail="Requested artifact not found")


if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
