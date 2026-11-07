/**
 * Single-card flow:
 * idle ("speak!") → recording (live script following) → checking (Vosk coverage)
 * → engine choice → loading → result / error.
 */

import { MicRecorder } from "./recorder.js?v=20261107";
import { LiveRecognizer } from "./liveRecognizer.js?v=20261107";
import { buildScript, classifyWord, computeCoverage, normalizeWord, scriptCursor, tokenize } from "./coverage.js?v=20261107";

const MIN_RECORDING_SEC = 1.5;
const MORPH_MS = 450;

const escapeHtml = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

const formatTime = (sec) => {
  const mins = String(Math.floor(sec / 60)).padStart(2, "0");
  const secs = (sec % 60).toFixed(1).padStart(4, "0");
  return `${mins}:${secs}`;
};

class SpeakToSingApp {
  constructor() {
    this.card = document.getElementById("stage");
    this.content = document.getElementById("stage-content");
    this.uploadLink = document.getElementById("upload-link");
    this.fileInput = document.getElementById("file-upload");
    this.hint = document.getElementById("stage-hint");

    this.sentences = [];
    this.scriptWords = [];
    this.scriptWordSet = new Set();
    this.coverageMap = [];
    this.hasEnvKey = false;
    this.hasVosk = false;

    this.audioFile = null;
    this.audioUrl = null;
    this.isMicTake = false;
    // Words heard live in earlier segments of the current take ("keep going").
    this.priorTokens = [];
    this.liveFinal = [];
    this.livePartial = "";
    // Words Vosk heard in the last server-side check of the current take.
    this.voskHeard = new Set();
    this.verdict = null;
    this.verdictError = "";

    this.singerName = "";
    this.apiKey = "";
    this.lastEngine = null;
    this.timerInterval = null;
    this.morphTimer = null;
    this.starting = false;

    this.recorder = new MicRecorder();
    this.live = new LiveRecognizer((finalWords, partialText, error) =>
      this._onLive(finalWords, partialText, error)
    );

    this.configLoaded = false;
    this._bindEvents();
    this._showIdle();
    this._loadConfig();
  }

  async _loadConfig() {
    try {
      const resp = await fetch("/api/config");
      if (!resp.ok) throw new Error(`Server returned ${resp.status}`);
      const cfg = await resp.json();
      this.sentences = cfg.sentences || [];
      this.scriptWords = buildScript(this.sentences);
      this.scriptWordSet = new Set(this.scriptWords.map((w) => w.norm));
      this.coverageMap = cfg.coverage_map || [];
      this.hasEnvKey = Boolean(cfg.has_env_elevenlabs_key);
      this.hasVosk = Boolean(cfg.has_vosk);
      this.configLoaded = true;
      if (this.card.dataset.state === "idle") this._showIdle();
    } catch (err) {
      console.error("Failed to fetch config:", err);
      this._setHint(`Couldn’t reach the server (${err.message}). Reload to try again.`);
    }
  }

  _bindEvents() {
    const activate = () => {
      if (this.starting || !this.configLoaded) return;
      const state = this.card.dataset.state;
      if (state === "idle") this._startRecording();
      else if (state === "recording") this._stopRecording();
    };
    this.card.addEventListener("click", (e) => {
      if (e.target.closest("a, button, input, label, audio")) return;
      activate();
    });
    this.card.addEventListener("keydown", (e) => {
      if ((e.key === "Enter" || e.key === " ") && e.target === this.card) {
        e.preventDefault();
        activate();
      }
    });

    this.fileInput.addEventListener("change", (e) => {
      const file = e.target.files?.[0];
      e.target.value = "";
      if (!file || !this.configLoaded) return;
      this._resetTake();
      this._setAudio(file);
      this.isMicTake = false;
      this._checkCoverage();
    });
  }

  /** Swap card content and animate the card's size between the old and new layout. */
  _morph(state, html, coverage = "") {
    const card = this.card;
    const from = card.getBoundingClientRect();

    card.dataset.state = state;
    card.dataset.coverage = coverage;
    card.tabIndex = state === "idle" || state === "recording" ? 0 : -1;
    card.setAttribute("role", state === "idle" || state === "recording" ? "button" : "region");
    this.content.innerHTML = html;
    this.content.classList.remove("fade-in");
    void this.content.offsetWidth;
    this.content.classList.add("fade-in");

    this.uploadLink.classList.toggle("hidden", state !== "idle");

    card.style.transition = "none";
    card.style.width = "";
    card.style.height = "";
    const to = card.getBoundingClientRect();
    card.style.width = `${from.width}px`;
    card.style.height = `${from.height}px`;
    void card.offsetWidth;
    card.style.transition = "";
    card.style.width = `${to.width}px`;
    card.style.height = `${to.height}px`;

    clearTimeout(this.morphTimer);
    this.morphTimer = setTimeout(() => {
      card.style.width = "";
      card.style.height = "";
    }, MORPH_MS + 50);
  }

  _setHint(text) {
    this.hint.textContent = text || "";
    this.hint.classList.toggle("hidden", !text);
  }

  _setAudio(file) {
    if (this.audioUrl) URL.revokeObjectURL(this.audioUrl);
    this.audioFile = file;
    this.audioUrl = URL.createObjectURL(file);
  }

  _resetTake() {
    this.priorTokens = [];
    this.liveFinal = [];
    this.livePartial = "";
    this.voskHeard = new Set();
    this.verdict = null;
    this.verdictError = "";
  }

  /** Script markup; `classOf(word)` optionally returns an extra class per word. */
  _scriptHtml(classOf = () => "") {
    return this.sentences
      .map((_, sIdx) => {
        const words = this.scriptWords
          .filter((w) => w.sentence === sIdx)
          .map((w) => `<span class="sw ${classOf(w)}" data-idx="${w.index}">${escapeHtml(w.display)}</span>`)
          .join(" ");
        return `<p class="script-line">${words}</p>`;
      })
      .join("");
  }

  // ---- States ----

  _showIdle(hint = "") {
    this._morph(
      "idle",
      `
      <h1 class="speak-title">speak!</h1>
      ${this.sentences.length ? `<p class="muted">Tap, then read this out loud:</p><div class="script">${this._scriptHtml()}</div>` : ""}
    `
    );
    this._setHint(hint);
  }

  async _startRecording({ append = false } = {}) {
    if (!append) this._resetTake();
    const live = this.hasVosk;

    this._morph(
      "recording",
      `
      <div class="rec-top">
        <span class="rec-dot"></span>
        <span id="rec-timer" class="rec-timer">${formatTime(this.recorder.elapsedSec)}</span>
        ${live ? `<span id="cov-counter" class="cov-counter"></span>` : ""}
      </div>
      <div id="script" class="script">${this._scriptHtml()}</div>
      ${live ? `<p id="cov-missing" class="missing"></p>` : ""}
      <canvas id="rec-canvas" class="rec-canvas" width="520" height="60"></canvas>
      <p class="muted small">${
        live
          ? "tap to stop"
          : "tap to stop · live script tracking is unavailable (server has no Vosk)"
      }</p>
    `
    );
    this._setHint(append ? "Keep going — say the missing words." : "");
    if (live) this._updateLive();

    this.starting = true;
    let liveOk = false;
    try {
      if (live) {
        liveOk = await this.live.connect();
        if (!liveOk) this._showLiveError("couldn’t connect to the server");
      }
      this.recorder.onFrame = liveOk ? (chunk, sr) => this.live.send(chunk, sr) : null;
      await this.recorder.start(document.getElementById("rec-canvas"), { append });
    } catch (err) {
      if (liveOk) this.live.stop();
      this._showIdle(`Microphone unavailable (${err.message}). Try uploading a file instead.`);
      return;
    } finally {
      this.starting = false;
    }

    const timerEl = document.getElementById("rec-timer");
    this.timerInterval = setInterval(() => {
      timerEl.textContent = formatTime(this.recorder.elapsedSec);
    }, 100);
  }

  _onLive(finalWords, partialText, error) {
    if (this.card.dataset.state !== "recording") return;
    if (error) {
      this._showLiveError(error);
      return;
    }
    this.liveFinal = finalWords.map(normalizeWord);
    this.livePartial = partialText;
    this._updateLive();
  }

  _showLiveError(message) {
    const missingEl = document.getElementById("cov-missing");
    if (missingEl) missingEl.textContent = `Live tracking unavailable (${message}) — your take will be checked after you stop.`;
  }

  /**
   * Re-color script words and the coverage border. Only confident final words count
   * toward coverage; the in-progress partial only moves the cursor.
   */
  _updateLive() {
    const finalTokens = [...this.priorTokens, ...this.liveFinal];
    const heard = new Set([...this.voskHeard, ...finalTokens]);
    const cursor = scriptCursor(this.scriptWords, [...finalTokens, ...tokenize(this.livePartial)]);
    const cov = computeCoverage(this.coverageMap, heard, this.scriptWordSet);

    this.content.querySelectorAll(".sw").forEach((el) => {
      const w = this.scriptWords[Number(el.dataset.idx)];
      const state = classifyWord(w, heard, cursor, cov);
      for (const cls of ["heard", "needed", "passed"]) el.classList.toggle(cls, state === cls);
      el.classList.toggle("cursor", w.index === cursor);
    });

    if (!this.coverageMap.length) return;
    const complete = cov.covered === cov.total;
    this.card.dataset.coverage = complete ? "ok" : "bad";

    const counter = document.getElementById("cov-counter");
    if (counter) counter.textContent = `sounds ${cov.covered}/${cov.total}`;
    const missingEl = document.getElementById("cov-missing");
    if (missingEl) {
      missingEl.textContent = complete ? "Enough to go — tap to stop." : `Still need: ${cov.missingWords.join(", ")}`;
    }
  }

  _stopRecording() {
    clearInterval(this.timerInterval);
    this.recorder.onFrame = null;
    this.live.stop();
    this.priorTokens.push(...this.liveFinal);
    this.liveFinal = [];
    this.livePartial = "";
    const { file, durationSec } = this.recorder.stop();

    if (durationSec < MIN_RECORDING_SEC) {
      this._showIdle("That was too short — try reading at least one full sentence.");
      return;
    }
    this._setAudio(file);
    this.isMicTake = true;
    this._checkCoverage();
  }

  async _checkCoverage() {
    this.verdict = null;
    this.verdictError = "";

    if (!this.hasVosk) {
      this._setHint("");
      this._showChoose();
      return;
    }

    this._morph("checking", `<div class="spinner" aria-label="Checking"></div>`);
    this._setHint("Checking your take…");

    try {
      const formData = new FormData();
      formData.append("audio", this.audioFile, this.audioFile.name || "recording.wav");
      const resp = await fetch("/api/coverage", { method: "POST", body: formData });
      const body = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new Error(body.detail || `Server returned ${resp.status}`);
      this.voskHeard = new Set((body.heard_words || []).map(normalizeWord));
      this.verdict = computeCoverage(this.coverageMap, this.voskHeard, this.scriptWordSet);
    } catch (err) {
      this.verdictError = err.message;
    }
    this._setHint("");
    this._showChoose();
  }

  _verdictHtml() {
    if (!this.hasVosk) {
      return `<p class="verdict muted">Local mode is unavailable on this server (Vosk missing) — ElevenLabs only.</p>`;
    }
    if (this.verdictError) {
      return `<p class="verdict muted">Couldn’t check your take (${escapeHtml(this.verdictError)}).</p>`;
    }
    if (!this.verdict) return "";
    const { covered, total, missingWords } = this.verdict;
    if (covered === total) {
      return `<p class="verdict ok">✓ All ${total} sounds covered</p>`;
    }
    // Same coloring as the live view, but from the final check (the whole script counts as passed).
    const scriptHtml = this._scriptHtml((w) =>
      classifyWord(w, this.voskHeard, Infinity, this.verdict)
    );
    return `
      <p class="verdict bad">${covered}/${total} sounds covered — Local mode will sound rougher.</p>
      <div class="script script-compact verdict-script">${scriptHtml}</div>
      <p class="missing">Still need: ${escapeHtml(missingWords.join(", "))}</p>
      ${this.isMicTake ? `<button type="button" id="btn-keep-going" class="btn">Keep going</button>` : ""}
    `;
  }

  _showChoose({ askKey = false } = {}) {
    const coverage = this.verdict ? (this.verdict.covered === this.verdict.total ? "ok" : "bad") : "";
    this._morph(
      "choose",
      `
      ${this._verdictHtml()}
      <audio class="preview" controls src="${this.audioUrl}"></audio>
      <input id="singer-name" class="text-input" type="text" placeholder="Singer name (optional)"
        value="${escapeHtml(this.singerName)}" maxlength="40" />
      <div class="engine-row">
        <button type="button" class="engine-btn" data-engine="elevenlabs">
          <strong>ElevenLabs</strong><small>Cloud voice clone</small>
        </button>
        <button type="button" class="engine-btn" data-engine="local" ${this.hasVosk ? "" : "disabled"}>
          <strong>Local</strong><small>${this.hasVosk ? "Offline, no credits" : "Unavailable (no Vosk)"}</small>
        </button>
      </div>
      <div id="key-row" class="key-row ${askKey ? "" : "hidden"}">
        <input id="api-key" class="text-input" type="password" autocomplete="off"
          placeholder="ElevenLabs API key" value="${escapeHtml(this.apiKey)}" />
        <button type="button" id="btn-key-go" class="btn">Generate</button>
      </div>
      <button type="button" id="btn-redo" class="link-btn">record again</button>
    `,
      coverage
    );

    document.getElementById("btn-keep-going")?.addEventListener("click", () =>
      this._startRecording({ append: true })
    );

    const nameInput = document.getElementById("singer-name");
    nameInput.addEventListener("input", () => (this.singerName = nameInput.value.trim()));

    const keyRow = document.getElementById("key-row");
    const keyInput = document.getElementById("api-key");
    keyInput.addEventListener("input", () => (this.apiKey = keyInput.value.trim()));

    const goEleven = () => {
      if (!this.hasEnvKey && !this.apiKey) {
        keyRow.classList.remove("hidden");
        keyInput.focus();
        return;
      }
      this._generate("elevenlabs");
    };

    this.content.querySelectorAll(".engine-btn").forEach((btn) =>
      btn.addEventListener("click", () => {
        if (btn.dataset.engine === "local") this._generate("local");
        else goEleven();
      })
    );
    document.getElementById("btn-key-go").addEventListener("click", goEleven);
    keyInput.addEventListener("keydown", (e) => e.key === "Enter" && goEleven());
    document.getElementById("btn-redo").addEventListener("click", () => this._showIdle());
  }

  async _generate(engine) {
    this.lastEngine = engine;
    this._morph("loading", `<div class="spinner" aria-label="Generating"></div>`);
    this._setHint("Building your voicebank — this takes about 30 seconds…");

    const formData = new FormData();
    formData.append("audio", this.audioFile, this.audioFile.name || "recording.wav");
    if (this.singerName) formData.append("character_name", this.singerName);
    formData.append("use_demo_mode", String(engine === "local"));
    formData.append("allow_fallback", "false");
    if (engine === "elevenlabs" && this.apiKey) formData.append("api_key", this.apiKey);

    try {
      const resp = await fetch("/api/voicebank/generate", { method: "POST", body: formData });
      const body = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        const detail = body.detail;
        const isEleven = detail && typeof detail === "object" && detail.code === "elevenlabs_failed";
        const message = (typeof detail === "object" ? detail?.message : detail) || `Server returned ${resp.status}`;
        this._showError(message, isEleven);
        return;
      }
      this._showResult(body);
    } catch (err) {
      this._showError(err.message, false);
    }
  }

  _showResult(data) {
    this._setHint("");
    this._morph(
      "result",
      `
      <p class="muted small">${escapeHtml(data.character_name)} sings</p>
      <h2 class="result-title">Daisy Bell</h2>
      <audio class="player" controls autoplay src="${escapeHtml(data.urls.daisy_bell_wav)}"></audio>
      <a class="btn btn-primary" href="${escapeHtml(data.urls.voicebank_zip)}" download>Download voicebank (.zip)</a>
      <button type="button" id="btn-restart" class="link-btn">start over</button>
    `
    );
    document.getElementById("btn-restart").addEventListener("click", () => this._showIdle());
  }

  _showError(message, isElevenLabsError) {
    this._setHint("");
    this._morph(
      "error",
      `
      <h2 class="error-title">${isElevenLabsError ? "ElevenLabs failed" : "Something went wrong"}</h2>
      <p class="error-msg">${escapeHtml(message)}</p>
      <div class="engine-row">
        <button type="button" id="btn-retry" class="btn">Try again</button>
        ${isElevenLabsError && this.hasVosk ? '<button type="button" id="btn-local" class="btn">Use local</button>' : ""}
      </div>
      <button type="button" id="btn-back" class="link-btn">back</button>
    `
    );
    document.getElementById("btn-retry").addEventListener("click", () => {
      // An invalid key is the most common failure, so let the user edit it before retrying.
      if (isElevenLabsError && !this.hasEnvKey) this._showChoose({ askKey: true });
      else this._generate(this.lastEngine);
    });
    document.getElementById("btn-local")?.addEventListener("click", () => this._generate("local"));
    document.getElementById("btn-back").addEventListener("click", () => this._showChoose());
  }
}

window.addEventListener("DOMContentLoaded", () => {
  new SpeakToSingApp();
});
