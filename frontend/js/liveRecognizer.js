/**
 * Live script tracking via the server's Vosk model: streams 16 kHz PCM16 over a WebSocket.
 */

const TARGET_RATE = 16000;
const CONNECT_TIMEOUT_MS = 3000;

export class LiveRecognizer {
  constructor(onUpdate) {
    this.onUpdate = onUpdate;
    this.ws = null;
    this.finalWords = [];
    this.partialText = "";
    this.leftover = new Float32Array(0);
  }

  /** Opens the socket; resolves true when live tracking is available. */
  connect() {
    this.finalWords = [];
    this.partialText = "";
    this.leftover = new Float32Array(0);

    return new Promise((resolve) => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${proto}://${location.host}/api/live`);
      ws.binaryType = "arraybuffer";
      const timer = setTimeout(() => {
        ws.close();
        resolve(false);
      }, CONNECT_TIMEOUT_MS);

      ws.onopen = () => {
        clearTimeout(timer);
        this.ws = ws;
        resolve(true);
      };
      ws.onerror = () => {
        clearTimeout(timer);
        resolve(false);
      };
      ws.onmessage = (e) => this._onMessage(e.data);
      ws.onclose = () => {
        if (this.ws === ws) this.ws = null;
      };
    });
  }

  _onMessage(raw) {
    let msg;
    try {
      msg = JSON.parse(raw);
    } catch {
      return;
    }
    if (msg.type === "final") {
      this.finalWords.push(...(msg.words || []).map((w) => w.word));
      this.partialText = "";
    } else if (msg.type === "partial") {
      this.partialText = msg.text || "";
    } else if (msg.type === "error") {
      this.onUpdate(this.finalWords, "", msg.message);
      return;
    }
    this.onUpdate(this.finalWords, this.partialText);
  }

  /** Accepts a Float32 mic chunk at `sampleRate` and streams it as 16 kHz PCM16. */
  send(chunk, sampleRate) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    const pcm = this._downsample(chunk, sampleRate);
    if (pcm.length) this.ws.send(pcm.buffer);
  }

  /** Flushes the recognizer; the final words still arrive through onUpdate. */
  stop() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.send("eof");
    this.ws = null;
  }

  _downsample(input, inRate) {
    let buf = input;
    if (this.leftover.length) {
      buf = new Float32Array(this.leftover.length + input.length);
      buf.set(this.leftover);
      buf.set(input, this.leftover.length);
    }
    const ratio = inRate / TARGET_RATE;
    const nOut = Math.floor(buf.length / ratio);
    const out = new Int16Array(nOut);
    for (let i = 0; i < nOut; i++) {
      // Box-filter average over each output window (cheap anti-aliasing).
      const s = Math.floor(i * ratio);
      const e = Math.min(buf.length, Math.floor((i + 1) * ratio));
      let sum = 0;
      for (let j = s; j < e; j++) sum += buf[j];
      const v = Math.max(-1, Math.min(1, sum / Math.max(1, e - s)));
      out[i] = v < 0 ? v * 0x8000 : v * 0x7fff;
    }
    this.leftover = buf.slice(Math.floor(nOut * ratio));
    return out;
  }
}
