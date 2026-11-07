# OpenUTAU Voicebank Studio — Speak to Sing

Turn 1–3 spoken sentences into a complete, ready-to-install **English ARPAbet (`EN ARPA+` & CV compatible) OpenUTAU voicebank**—and immediately hear your newly created voicebank sing **"Daisy Bell (Bicycle Built for Two)"** right in the browser!

---

## Features

1. **Guided 1–3 Sentence Teleprompter & Live WebAudio Recorder**
   - Record directly in your browser with real-time `<canvas>` waveform visualization (encodes clean 44.1kHz 16-bit PCM WAV client-side) or upload any `.wav` / `.mp3` / `.webm` voice clip.
2. **Dual Voice Cloning Engine**
   - **ElevenLabs Instant Voice Cloning + Batched Timestamp Alignment**: Clones your voice via `POST /v1/voices/add`, synthesizes 48 ARPAbet phoneme carriers in 4 compact batches via `POST /v1/text-to-speech/{id}/with-timestamps`, slices each phoneme using exact character timestamps, and **automatically deletes the temporary voice slot** (`DELETE /v1/voices/{id}`) so you never run out of account slots.
   - **Zero-Credit Local Acoustic Timbre Transfer Mode**: Extracts your median F0, vocal tract formant scale, and harmonic roll-off locally to synthesize all 48 ARPAbet samples without spending API credits.
3. **WORLD Vocoder (`pyworld`) Pitch Normalization & Acoustic `oto.ini` Alignment**
   - Flattens the pitch of every voiced phoneme sample to a steady **C4 (261.63 Hz)** while preserving unvoiced consonant bursts/fricatives and your vocal tract spectral envelope.
   - Exports strictly as **44,100 Hz, 16-bit, Mono PCM `.wav`** and calculates UTAU `oto.ini` parameters (`Offset`, `Overlap`, `Preutterance`, `Consonant`, `Cutoff`).
4. **In-Browser "Daisy Bell" UTAU Performance & `.ustx` Exporter**
   - Includes a built-in WORLD vocoder UTAU resampler that renders *"Daisy Bell (Bicycle Built for Two)"* using your newly generated `.wav` files and `oto.ini` rules, with a synchronized piano-roll & karaoke lyric ribbon.
   - Exports your `<SingerName>_OpenUTAU.zip` voicebank alongside `Daisy_Bell.ustx` (native OpenUTAU project) and `Daisy_Bell.ust` (classic UTAU sequence).

---

## Quick Start

```bash
# 1. Activate the virtual environment (already created in .venv)
source .venv/bin/activate

# 2. (Optional) Configure your ElevenLabs API key in .env (or paste it directly in the Web UI)
cp .env.example .env

# 3. Launch the unified FastAPI + Vanilla JS Studio server
.venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open **http://localhost:8000** in your browser!

On first start the server downloads the Vosk `en-us` model (~40 MB). Without Vosk, the server still runs but Local mode, live script tracking (streamed to Vosk over the `/api/live` WebSocket) and the take-coverage check are disabled.

---

## Loading Your Voicebank in OpenUTAU

1. Click **Download voicebank (.zip)** in the Studio UI (`Daisy_Bell.ustx` is included inside the zip).
2. Open **OpenUTAU** and select `Tools -> Install Singer...`, then choose your downloaded `<SingerName>_OpenUTAU.zip` archive.
3. Select `File -> Open...` and open `Daisy_Bell.ustx` (or create a new track, select your singer, and use either the **EN ARPA+** / **EN ARPA** phonemizer or the **Default** phonemizer).
