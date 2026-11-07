/**
 * WebAudio microphone recorder: captures mono PCM, draws a live waveform, encodes 16-bit WAV.
 */

export class MicRecorder {
  constructor() {
    this.isRecording = false;
    this.audioCtx = null;
    this.mediaStream = null;
    this.analyser = null;
    this.processor = null;
    this.recordedChunks = [];
    this.recordingSr = 44100;
    this.startTime = 0;
    this.priorSec = 0;
    this.animFrame = null;
    this.canvas = null;
    // Optional (chunk: Float32Array, sampleRate: number) => void, called for each captured block.
    this.onFrame = null;
  }

  /** Total recorded time, including earlier appended segments. */
  get elapsedSec() {
    return this.priorSec + (this.isRecording ? (performance.now() - this.startTime) / 1000 : 0);
  }

  /** Starts recording. With append=true, new audio is added to the previous take. */
  async start(canvas, { append = false } = {}) {
    if (!append) {
      this.recordedChunks = [];
      this.priorSec = 0;
    }
    this.canvas = canvas;
    this.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });

    try {
      this.audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 44100 });
      if (append && this.recordedChunks.length && this.audioCtx.sampleRate !== this.recordingSr) {
        throw new Error(`sample rate changed (${this.recordingSr} → ${this.audioCtx.sampleRate} Hz)`);
      }
      this.recordingSr = this.audioCtx.sampleRate;
      const source = this.audioCtx.createMediaStreamSource(this.mediaStream);

      this.analyser = this.audioCtx.createAnalyser();
      this.analyser.fftSize = 1024;
      source.connect(this.analyser);

      this.processor = this.audioCtx.createScriptProcessor(4096, 1, 1);
      this.processor.onaudioprocess = (e) => {
        if (!this.isRecording) return;
        const chunk = new Float32Array(e.inputBuffer.getChannelData(0));
        this.recordedChunks.push(chunk);
        if (this.onFrame) this.onFrame(chunk, this.recordingSr);
      };
      source.connect(this.processor);
      this.processor.connect(this.audioCtx.destination);
    } catch (err) {
      this._teardown();
      throw err;
    }

    this.isRecording = true;
    this.startTime = performance.now();
    this._animateWaveform();
  }

  /** Stops recording and returns { file, durationSec } for the whole take so far. */
  stop() {
    const durationSec = this.elapsedSec;
    this._teardown();
    this.priorSec = durationSec;
    const wavBlob = this._encodeWavFromChunks(this.recordedChunks, this.recordingSr);
    const file = new File([wavBlob], "mic_recording.wav", { type: "audio/wav" });
    return { file, durationSec };
  }

  _teardown() {
    this.isRecording = false;
    cancelAnimationFrame(this.animFrame);
    if (this.processor) this.processor.disconnect();
    if (this.mediaStream) this.mediaStream.getTracks().forEach((t) => t.stop());
    if (this.audioCtx && this.audioCtx.state !== "closed") this.audioCtx.close();
  }

  _animateWaveform() {
    if (!this.isRecording || !this.analyser || !this.canvas) return;
    const ctx = this.canvas.getContext("2d");
    const bufLen = this.analyser.frequencyBinCount;
    const timeData = new Uint8Array(bufLen);
    this.analyser.getByteTimeDomainData(timeData);

    const w = this.canvas.width;
    const h = this.canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
    ctx.beginPath();
    const sliceWidth = w / bufLen;
    for (let i = 0, x = 0; i < bufLen; i++, x += sliceWidth) {
      const y = ((timeData[i] / 128.0) * h) / 2;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    this.animFrame = requestAnimationFrame(() => this._animateWaveform());
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
