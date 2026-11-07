/**
 * Main OpenUTAU Voicebank Studio Frontend Controller.
 */

import { VoiceRecorderController } from "./recorder.js";
import { DaisyBellPlayerController } from "./daisyBellPlayer.js";
import { OtoInspectorController } from "./otoInspector.js";

class OpenUtauStudioApp {
  constructor() {
    this.readyAudioFile = null;
    this.avatarFile = null;

    this.otoInspector = new OtoInspectorController();
    this.daisyPlayer = new DaisyBellPlayerController((sampleId, autoPlay) => {
      this.otoInspector.selectSampleById(sampleId, autoPlay);
    });
    this.recorder = new VoiceRecorderController((audioBlob) => {
      this.readyAudioFile = audioBlob;
      document.getElementById("error-banner").classList.add("hidden");
    });

    this._bindFormEvents();
    this._loadStudioConfig();
  }

  async _loadStudioConfig() {
    try {
      const resp = await fetch("/api/config");
      const cfg = await resp.json();
      this.recorder.setSentences(cfg.sentences || []);

      if (cfg.has_env_elevenlabs_key) {
        document.getElementById("env-key-badge").classList.remove("hidden");
      }
      if (!cfg.has_pyworld) {
        document.getElementById("badge-pyworld").textContent = "Resampler: C4 (261.63 Hz)";
      }
    } catch (err) {
      console.error("Failed to fetch studio config:", err);
    }
  }

  _bindFormEvents() {
    const charNameInput = document.getElementById("character-name");
    const avatarInput = document.getElementById("avatar-upload");
    const avatarPreview = document.getElementById("avatar-preview-box");

    charNameInput.addEventListener("input", () => {
      if (!this.avatarFile) {
        const initials =
          charNameInput.value
            .trim()
            .split(/[\s_]+/)
            .map((w) => w[0]?.toUpperCase() || "")
            .join("")
            .slice(0, 2) || "UV";
        avatarPreview.textContent = initials;
      }
    });

    avatarInput.addEventListener("change", (e) => {
      const file = e.target.files?.[0];
      if (!file) return;
      this.avatarFile = file;
      const url = URL.createObjectURL(file);
      avatarPreview.innerHTML = `<img src="${url}" alt="Avatar Preview" />`;
    });

    // Mode radio cards
    const cardEleven = document.getElementById("card-mode-eleven");
    const cardDemo = document.getElementById("card-mode-demo");
    const elevenBox = document.getElementById("elevenlabs-settings");

    document.querySelectorAll('input[name="engine_mode"]').forEach((radio) => {
      radio.addEventListener("change", () => {
        const isDemo = radio.value === "demo";
        cardEleven.classList.toggle("active", !isDemo);
        cardDemo.classList.toggle("active", isDemo);
        elevenBox.classList.toggle("hidden", isDemo);
      });
    });

    const form = document.getElementById("generate-form");
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      this._handleGenerateSubmit();
    });
  }

  async _handleGenerateSubmit() {
    const errorBanner = document.getElementById("error-banner");
    errorBanner.classList.add("hidden");

    if (!this.readyAudioFile) {
      errorBanner.textContent = "Please record your voice using the microphone or upload an audio file first.";
      errorBanner.classList.remove("hidden");
      return;
    }

    const btnGenerate = document.getElementById("btn-generate");
    const progressBox = document.getElementById("pipeline-progress");
    const progressFill = document.getElementById("pipeline-progress-fill");
    const stageText = document.getElementById("pipeline-stage-text");
    const stepItems = document.querySelectorAll("#pipeline-step-list li");

    btnGenerate.disabled = true;
    progressBox.classList.remove("hidden");

    const charName = document.getElementById("character-name").value.trim() || "Aria_UTAU";
    const authorName = document.getElementById("author-name").value.trim() || "GreatUniHack Creator";
    const engineMode = document.querySelector('input[name="engine_mode"]:checked')?.value || "elevenlabs";
    const apiKey = document.getElementById("elevenlabs-key").value.trim();
    const autoCleanup = document.getElementById("auto-cleanup").checked;

    const formData = new FormData();
    formData.append("audio", this.readyAudioFile, this.readyAudioFile.name || "recording.wav");
    formData.append("character_name", charName);
    formData.append("author_name", authorName);
    formData.append("use_demo_mode", String(engineMode === "demo"));
    formData.append("auto_cleanup_voice", String(autoCleanup));
    if (apiKey) {
      formData.append("api_key", apiKey);
    }
    if (this.avatarFile) {
      formData.append("avatar", this.avatarFile);
    }

    // Animate pipeline stepper while backend synthesizes and packages
    const stages = [
      { step: 1, pct: 16, text: "Decoding 44.1kHz reference audio & analyzing F0/formants..." },
      { step: 2, pct: 38, text: "Synthesizing 48 ARPAbet phoneme carriers in your vocal timbre..." },
      { step: 3, pct: 58, text: "Flattening pitch contour to C4 (261.63 Hz) via WORLD vocoder..." },
      { step: 4, pct: 74, text: "Detecting consonant-vowel boundaries & building oto.ini..." },
      { step: 5, pct: 88, text: "Resampling 'Daisy Bell (Bicycle Built for Two)' with voicebank..." },
      { step: 6, pct: 95, text: "Packaging OpenUTAU .zip archive & Daisy_Bell.ustx..." },
    ];

    let curStageIdx = 0;
    const updateStepUI = (idx) => {
      const st = stages[idx];
      if (!st) return;
      stageText.textContent = st.text;
      progressFill.style.width = `${st.pct}%`;
      stepItems.forEach((li) => {
        const sNum = Number(li.dataset.step);
        li.classList.toggle("done", sNum < st.step);
        li.classList.toggle("active", sNum === st.step);
      });
    };

    updateStepUI(0);
    const stepTimer = setInterval(() => {
      if (curStageIdx < stages.length - 1) {
        curStageIdx++;
        updateStepUI(curStageIdx);
      }
    }, 900);

    try {
      const resp = await fetch("/api/voicebank/generate", {
        method: "POST",
        body: formData,
      });

      clearInterval(stepTimer);

      if (!resp.ok) {
        const errBody = await resp.json().catch(() => ({}));
        throw new Error(errBody.detail || `Server returned ${resp.status}`);
      }

      const data = await resp.json();
      progressFill.style.width = "100%";
      stageText.textContent = "Voicebank & Daisy Bell performance ready!";
      stepItems.forEach((li) => {
        li.classList.remove("active");
        li.classList.add("done");
      });

      this._renderResults(data);
    } catch (err) {
      clearInterval(stepTimer);
      errorBanner.textContent = `Generation error: ${err.message}`;
      errorBanner.classList.remove("hidden");
    } finally {
      btnGenerate.disabled = false;
    }
  }

  _renderResults(data) {
    const resultsSection = document.getElementById("results-section");
    resultsSection.classList.remove("hidden");

    document.getElementById("result-avatar").src = data.urls.avatar_bmp;
    document.getElementById("result-singer-name").textContent = data.character_name;
    const vp = data.voice_profile || {};
    const baseNote = vp.base_note_name || "C4";
    const baseHz = vp.base_note_hz || 261.6;
    document.getElementById("result-mode-badge").textContent =
      data.mode_used === "elevenlabs_ivc"
        ? `ElevenLabs IVC + WORLD ${baseNote}`
        : `Direct Voice Corpus + WORLD ${baseNote}`;
    document.getElementById("result-mode-detail").textContent = data.mode_detail;
    document.getElementById("btn-play-sample").innerHTML = `&#9654; Play Sample (${baseNote})`;

    const pillsEl = document.getElementById("telemetry-pills");
    pillsEl.innerHTML = `
      <div class="telemetry-pill">Your Speaking F0: <strong>${vp.median_f0_hz || 175} Hz</strong></div>
      <div class="telemetry-pill">Voicebank Key: <strong>${baseNote} (${baseHz} Hz)</strong></div>
      <div class="telemetry-pill">Samples: <strong>${data.sample_count} WAVs</strong></div>
      <div class="telemetry-pill">oto.ini Aliases: <strong>${data.oto_alias_count}</strong></div>
    `;

    // Populate Download Links
    document.getElementById("dl-voicebank-zip").href = data.urls.voicebank_zip;
    document.getElementById("dl-daisy-ustx").href = data.urls.daisy_bell_ustx;
    document.getElementById("dl-daisy-ust").href = data.urls.daisy_bell_ust;
    document.getElementById("dl-daisy-wav").href = data.urls.daisy_bell_wav;
    document.getElementById("dl-oto-ini").href = data.urls.oto_ini;

    // Load Daisy Bell track & lyrics
    this.daisyPlayer.loadTrack(data.urls.daisy_bell_wav, data.daisy_bell_lyrics);

    // Load 48 phoneme samples into the oto.ini inspector
    this.otoInspector.setSamples(data.samples);

    // Smooth scroll to the studio results
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

window.addEventListener("DOMContentLoaded", () => {
  new OpenUtauStudioApp();
});
