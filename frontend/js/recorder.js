/**
 * Teleprompter & WebAudio Microphone WAV Recorder + Synthetic Test Voice Generator.
 */

export class VoiceRecorderController {
  constructor(onAudioReady) {
    this.onAudioReady = onAudioReady;
    this.sentences = [];
    this.selectedMode = "all";

    this.isRecording = false;
    this.audioCtx = null;
    this.mediaStream = null;
    this.analyser = null;
    this.processor = null;
    this.recordedChunks = [];
    this.recordingSr = 44100;
    this.startTime = 0;
    this.timerInterval = null;
    this.animFrame = null;

    this.canvas = document.getElementById("mic-waveform");
    this.ctx = this.canvas.getContext("2d");
    this.btnRecord = document.getElementById("btn-record");
    this.fileInput = document.getElementById("file-upload");
    this.statusDot = document.getElementById("rec-status-dot");
    this.statusText = document.getElementById("rec-status-text");
    this.timerEl = document.getElementById("rec-timer");
    this.previewBar = document.getElementById("recorded-preview-bar");
    this.sourceBadge = document.getElementById("recorded-source-badge");
    this.filenameEl = document.getElementById("recorded-filename");
    this.audioEl = document.getElementById("recorded-audio-el");

    this._bindEvents();
    this._drawIdleWaveform();
  }

  setSentences(sentences) {
    this.sentences = sentences || [];
    this._renderTeleprompter();
  }

  _bindEvents() {
    const tabContainer = document.getElementById("sentence-tabs");
    tabContainer.addEventListener("click", (e) => {
      const btn = e.target.closest(".tab-btn");
      if (!btn) return;
      tabContainer.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      this.selectedMode = btn.dataset.mode;
      this._renderTeleprompter();
    });

    this.btnRecord.addEventListener("click", () => {
      if (this.isRecording) {
        this.stopRecording();
      } else {
        this.startRecording();
      }
    });

    this.fileInput.addEventListener("change", (e) => {
      const file = e.target.files?.[0];
      if (file) {
        this._setReadyAudio(file, file.name, "Uploaded File");
      }
    });
  }

  _renderTeleprompter() {
    const box = document.getElementById("teleprompter-box");
    if (!this.sentences.length) return;

    const items =
      this.selectedMode === "all"
        ? this.sentences
        : [this.sentences[Number(this.selectedMode)] || this.sentences[0]];

    box.innerHTML = items
      .map(
        (s) => `
      <div class="prompt-item">
        <div class="prompt-item-title">${s.title}</div>
        <div class="prompt-item-text">&ldquo;${s.text}&rdquo;</div>
        <div class="prompt-item-focus">${s.focus}</div>
      </div>
    `
      )
      .join("");
  }

  async startRecording() {
    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });

      this.audioCtx = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 44100,
      });
      this.recordingSr = this.audioCtx.sampleRate;
      const source = this.audioCtx.createMediaStreamSource(this.mediaStream);

      this.analyser = this.audioCtx.createAnalyser();
      this.analyser.fftSize = 1024;
      source.connect(this.analyser);

      this.processor = this.audioCtx.createScriptProcessor(4096, 1, 1);
      this.recordedChunks = [];
      this.processor.onaudioprocess = (e) => {
        if (!this.isRecording) return;
        const input = e.inputBuffer.getChannelData(0);
        this.recordedChunks.push(new Float32Array(input));
      };

      source.connect(this.processor);
      this.processor.connect(this.audioCtx.destination);

      this.isRecording = true;
      this.startTime = performance.now();
      this.btnRecord.classList.add("is-recording");
      this.btnRecord.innerHTML = `<span>&#9632;</span> Stop Recording`;
      this.statusDot.className = "status-dot recording";
      this.statusText.textContent = "Recording live microphone... Speak the sentences above";

      this.timerInterval = setInterval(() => {
        const elapsed = (performance.now() - this.startTime) / 1000;
        const mins = String(Math.floor(elapsed / 60)).padStart(2, "0");
        const secs = (elapsed % 60).toFixed(1).padStart(4, "0");
        this.timerEl.textContent = `${mins}:${secs}`;
      }, 100);

      this._animateLiveWaveform();
    } catch (err) {
      alert(
        `Microphone access was unavailable (${err.message}). You can click "Upload Audio" or "Load Test Voice Sample" instead!`
      );
    }
  }

  stopRecording() {
    if (!this.isRecording) return;
    this.isRecording = false;
    clearInterval(this.timerInterval);
    cancelAnimationFrame(this.animFrame);

    if (this.processor) this.processor.disconnect();
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((t) => t.stop());
    }
    if (this.audioCtx && this.audioCtx.state !== "closed") {
      this.audioCtx.close();
    }

    this.btnRecord.classList.remove("is-recording");
    this.btnRecord.innerHTML = `<span class="btn-icon">&#9679;</span> Start Recording`;

    const wavBlob = this._encodeWavFromChunks(this.recordedChunks, this.recordingSr);
    const file = new File([wavBlob], "mic_recording.wav", { type: "audio/wav" });
    this._setReadyAudio(file, "mic_recording.wav", "Mic Recording");
  }

  _setReadyAudio(fileBlob, labelName, sourceType) {
    this.statusDot.className = "status-dot ready";
    this.statusText.textContent = `Voice sample ready (${sourceType})`;
    this.sourceBadge.textContent = sourceType;
    this.filenameEl.textContent = labelName;

    const url = URL.createObjectURL(fileBlob);
    this.audioEl.src = url;
    this.previewBar.classList.remove("hidden");

    this._drawStaticReadyWaveform();
    if (this.onAudioReady) {
      this.onAudioReady(fileBlob);
    }
  }

  _animateLiveWaveform() {
    if (!this.isRecording || !this.analyser) return;
    const bufLen = this.analyser.frequencyBinCount;
    const timeData = new Uint8Array(bufLen);
    this.analyser.getByteTimeDomainData(timeData);

    const w = this.canvas.width;
    const h = this.canvas.height;
    this.ctx.fillStyle = "#070b12";
    this.ctx.fillRect(0, 0, w, h);

    // Center reference line
    this.ctx.strokeStyle = "rgba(56, 189, 248, 0.15)";
    this.ctx.beginPath();
    this.ctx.moveTo(0, h / 2);
    this.ctx.lineTo(w, h / 2);
    this.ctx.stroke();

    // Oscilloscope trace
    this.ctx.lineWidth = 2.2;
    this.ctx.strokeStyle = "#38bdf8";
    this.ctx.beginPath();
    const sliceWidth = w / bufLen;
    let x = 0;
    for (let i = 0; i < bufLen; i++) {
      const v = timeData[i] / 128.0;
      const y = (v * h) / 2;
      if (i === 0) this.ctx.moveTo(x, y);
      else this.ctx.lineTo(x, y);
      x += sliceWidth;
    }
    this.ctx.stroke();

    this.animFrame = requestAnimationFrame(() => this._animateLiveWaveform());
  }

  _drawIdleWaveform() {
    const w = this.canvas.width;
    const h = this.canvas.height;
    this.ctx.fillStyle = "#070b12";
    this.ctx.fillRect(0, 0, w, h);

    this.ctx.strokeStyle = "rgba(56, 189, 248, 0.25)";
    this.ctx.lineWidth = 1.5;
    this.ctx.beginPath();
    for (let x = 0; x < w; x++) {
      const y = h / 2 + Math.sin(x * 0.03) * 8 * Math.sin((x / w) * Math.PI);
      if (x === 0) this.ctx.moveTo(x, y);
      else this.ctx.lineTo(x, y);
    }
    this.ctx.stroke();
  }

  _drawStaticReadyWaveform() {
    const w = this.canvas.width;
    const h = this.canvas.height;
    this.ctx.fillStyle = "#070b12";
    this.ctx.fillRect(0, 0, w, h);

    const bars = 90;
    const barW = w / bars;
    for (let i = 0; i < bars; i++) {
      const norm = Math.sin((i / bars) * Math.PI);
      const amp = (0.2 + 0.65 * Math.abs(Math.sin(i * 0.65 + 1.2)) * norm) * (h * 0.42);
      this.ctx.fillStyle = i % 2 === 0 ? "#10b981" : "#38bdf8";
      this.ctx.fillRect(i * barW + 1, h / 2 - amp, Math.max(2, barW - 2), amp * 2);
    }
  }

  _encodeWavFromChunks(chunks, sampleRate) {
    const totalLength = chunks.reduce((acc, c) => acc + c.length, 0);
    const samples = new Float32Array(totalLength);
    let offset = 0;
    for (const c of chunks) {
      samples.set(c, offset);
      offset += c.length;
    }
    return this._encodePcm16Wav(samples, sampleRate);
  }

  _generateSyntheticReferenceWav() {
    const sr = 44100;
    const durSec = 6.0;
    const total = sr * durSec;
    const samples = new Float32Array(total);
    // Synthesize a natural-sounding spoken vocal contour (~195 Hz median F0 with formant resonances)
    let phase = 0;
    for (let i = 0; i < total; i++) {
      const t = i / sr;
      const syllableEnv = 0.55 + 0.45 * Math.sin(2 * Math.PI * 2.4 * t);
      const f0 = 195 + 25 * Math.sin(2 * Math.PI * 0.7 * t) + 8 * Math.cos(2 * Math.PI * 1.9 * t);
      phase += (2 * Math.PI * f0) / sr;
      let val =
        Math.sin(phase) +
        0.62 * Math.sin(2 * phase + 0.3) +
        0.41 * Math.sin(3 * phase + 0.7) +
        0.25 * Math.sin(4 * phase + 1.1) +
        0.15 * Math.sin(5 * phase) +
        0.09 * Math.sin(6 * phase);
      samples[i] = val * 0.24 * syllableEnv;
    }
    this.timerEl.textContent = "00:06.0";
    return this._encodePcm16Wav(samples, sr);
  }

  _encodePcm16Wav(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    const writeStr = (pos, str) => {
      for (let i = 0; i < str.length; i++) {
        view.setUint8(pos + i, str.charCodeAt(i));
      }
    };

    writeStr(0, "RIFF");
    view.setUint32(4, 36 + samples.length * 2, true);
    writeStr(8, "WAVE");
    writeStr(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // Mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeStr(36, "data");
    view.setUint32(40, samples.length * 2, true);

    let peak = 0;
    for (let i = 0; i < samples.length; i++) {
      const a = Math.abs(samples[i]);
      if (a > peak) peak = a;
    }
    const scale = peak > 0.68 ? 0.68 / peak : 1.0;

    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
      const s = Math.max(-0.98, Math.min(0.98, samples[i] * scale));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }

    return new Blob([buffer], { type: "audio/wav" });
  }
}
