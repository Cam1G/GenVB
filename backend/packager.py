"""
OpenUTAU & Classic UTAU Voicebank Packager:
Generates:
- character.txt (standard UTAU character definition)
- character.yaml (OpenUTAU metadata + EN ARPA / Default phonemizer settings)
- oto.ini (full ARPAbet + CV fallback alias timings)
- prefix.map (pitch map for OpenUTAU)
- avatar.bmp (custom uploaded image or procedural vocal-synth avatar)
- readme.txt (usage instructions for OpenUTAU)
- Daisy_Bell.ustx & Daisy_Bell.ust (project files)
- Zips the complete voicebank directory for 1-click OpenUTAU installation.
"""

from __future__ import annotations

import io
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from PIL import Image, ImageDraw, ImageFont

from .daisy_bell import generate_classic_ust, generate_openutau_ustx


def sanitize_folder_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_\-\s]", "", name or "MyVoicebank").strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned or "OpenUTAU_Voice"


def generate_avatar_bmp(
    output_path: Path,
    character_name: str,
    avatar_bytes: Optional[bytes] = None,
) -> None:
    """
    Create a 100x100 24-bit BMP avatar as expected by UTAU and OpenUTAU.
    If the user uploaded an image, center-crop and resize it; otherwise generate a
    sleek geometric vocal-synth avatar with the character's initials.
    """
    size = (100, 100)
    if avatar_bytes:
        try:
            with Image.open(io.BytesIO(avatar_bytes)) as im:
                im = im.convert("RGB")
                w, h = im.size
                side = min(w, h)
                left = (w - side) // 2
                top = (h - side) // 2
                im = im.crop((left, top, left + side, top + side)).resize(size, Image.Resampling.LANCZOS)
                im.save(str(output_path), format="BMP")
                return
        except Exception:
            pass

    # Procedural neon vocal-synth avatar
    img = Image.new("RGB", size, (15, 23, 42))
    draw = ImageDraw.Draw(img)

    # Gradient background bars
    for y in range(100):
        r = int(18 + (y / 100.0) * 45)
        g = int(28 + (y / 100.0) * 140)
        b = int(68 + (y / 100.0) * 165)
        draw.line([(0, y), (99, y)], fill=(r, g, b))

    # Waveform motif across center
    import math
    for x in range(6, 94, 4):
        amp = int(14 * math.sin((x / 94.0) * math.pi) * abs(math.cos(x * 0.25)))
        draw.line([(x, 76 - amp), (x, 76 + amp)], fill=(125, 249, 255), width=2)

    # Border frame
    draw.rectangle([2, 2, 97, 97], outline=(56, 189, 248), width=2)

    # Initials badge
    initials = "".join([w[0].upper() for w in (character_name or "UV").split() if w])[:2] or "UV"
    font = ImageFont.load_default()
    draw.ellipse([26, 14, 74, 62], fill=(15, 23, 42), outline=(125, 249, 255), width=2)
    bbox = draw.textbbox((0, 0), initials, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((50 - tw // 2, 38 - th // 2), initials, fill=(248, 250, 252), font=font)

    img.save(str(output_path), format="BMP")


def package_openutau_voicebank(
    session_dir: Path,
    voicebank_dir: Path,
    character_name: str,
    author_name: str,
    oto_lines: List[str],
    voice_profile: Dict[str, Any],
    mode_detail: str,
    avatar_bytes: Optional[bytes] = None,
) -> Dict[str, Path]:
    """
    Write all OpenUTAU voicebank configuration files into `voicebank_dir`,
    create `Daisy_Bell.ustx` & `Daisy_Bell.ust` in `session_dir`, and build the
    ready-to-install `<folder_name>.zip` archive.
    """
    safe_folder = sanitize_folder_name(character_name)
    display_name = (character_name or "Custom AI Singer").strip()
    display_author = (author_name or "GreatUniHack Creator").strip()

    # 1. avatar.bmp
    avatar_path = voicebank_dir / "avatar.bmp"
    generate_avatar_bmp(avatar_path, display_name, avatar_bytes)

    # 2. character.txt (CRLF for strict UTAU/OpenUTAU parser compatibility)
    char_txt_lines = [
        f"name={display_name}",
        "image=avatar.bmp",
        f"author={display_author}",
        "sample=01_aa.wav",
        "version=1.0 (ARPAbet + CV OpenUTAU)",
        "web=https://github.com/stakira/OpenUtau",
    ]
    (voicebank_dir / "character.txt").write_text("\r\n".join(char_txt_lines) + "\r\n", encoding="utf-8")

    # 3. character.yaml (OpenUTAU extended configuration)
    char_yaml = {
        "name": display_name,
        "singer_type": "utau",
        "text_file_encoding": "utf-8",
        "image": "avatar.bmp",
        "author": display_author,
        "version": "1.0",
        "default_phonemizer": "OpenUtau.Plugin.Builtin.ArpasingPlusPhonemizer",
        "subbanks": [
            {
                "color": "",
                "prefix": "",
                "suffix": "",
                "tone_ranges": ["C1-B7"],
            }
        ],
    }
    (voicebank_dir / "character.yaml").write_text(
        yaml.safe_dump(char_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    # 4. oto.ini
    (voicebank_dir / "oto.ini").write_text("\r\n".join(oto_lines) + "\r\n", encoding="utf-8")

    # 5. prefix.map (Covers MIDI notes C1..B7 with empty prefix/suffix)
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    prefix_lines = []
    for octave in range(7, 0, -1):
        for n_name in reversed(note_names):
            prefix_lines.append(f"{n_name}{octave}\t\t")
    (voicebank_dir / "prefix.map").write_text("\r\n".join(prefix_lines) + "\r\n", encoding="utf-8")

    # 6. readme.txt
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    base_note_name = voice_profile.get("base_note_name", "C4")
    base_note_hz = voice_profile.get("base_note_hz", 261.63)
    semitone_shift = int(voice_profile.get("semitone_shift", 0))
    readme_text = (
        f"=== {display_name} — OpenUTAU Voicebank ===\r\n"
        f"Created by: {display_author}\r\n"
        f"Generated: {timestamp}\r\n"
        f"Synthesis Engine: {mode_detail}\r\n"
        f"Reference Pitch: {base_note_name} ({base_note_hz} Hz, WORLD Vocoder Normalized)\r\n"
        f"Detected User Median Pitch: {voice_profile.get('median_f0_hz', 180.0)} Hz\r\n\r\n"
        "HOW TO USE IN OPENUTAU:\r\n"
        "1. Open OpenUTAU -> Tools -> Install Singer... and select this .zip file.\r\n"
        "2. Select this singer on any track.\r\n"
        "3. Works out-of-the-box with 'EN ARPA+' / 'EN ARPA' OR 'Default' phonemizers!\r\n"
        "4. Open the included 'Daisy_Bell.ustx' project file to hear or edit 'Daisy Bell'!\r\n"
    )
    (voicebank_dir / "readme.txt").write_text(readme_text, encoding="utf-8")

    # 7. Generate Daisy_Bell.ustx and Daisy_Bell.ust
    ustx_content = generate_openutau_ustx(display_name, safe_folder, semitone_shift=semitone_shift)
    ustx_path = session_dir / "Daisy_Bell.ustx"
    ustx_path.write_text(ustx_content, encoding="utf-8")

    ust_content = generate_classic_ust(display_name, semitone_shift=semitone_shift)
    ust_path = session_dir / "Daisy_Bell.ust"
    ust_path.write_text(ust_content, encoding="utf-8")

    # Also include a copy of Daisy_Bell.ustx & Daisy_Bell.ust inside the voicebank folder for convenience
    (voicebank_dir / "Daisy_Bell.ustx").write_text(ustx_content, encoding="utf-8")
    (voicebank_dir / "Daisy_Bell.ust").write_text(ust_content, encoding="utf-8")

    # 8. Create the OpenUTAU voicebank .zip archive
    zip_filename = f"{safe_folder}_OpenUTAU.zip"
    zip_path = session_dir / zip_filename
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(voicebank_dir.iterdir()):
            if file_path.is_file():
                arcname = f"{safe_folder}/{file_path.name}"
                zf.write(file_path, arcname=arcname)

    return {
        "zip_path": zip_path,
        "zip_filename": zip_filename,
        "ustx_path": ustx_path,
        "ust_path": ust_path,
        "avatar_path": avatar_path,
    }
