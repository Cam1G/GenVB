/**
 * Synchronized "Daisy Bell (Bicycle Built for Two)" Vocal Synth Player & Piano-Roll Visualizer.
 */

const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

function midiToNoteName(midi) {
  const octave = Math.floor(midi / 12) - 1;
  return `${NOTE_NAMES[midi % 12]}${octave}`;
}

export class DaisyBellPlayerController {
  constructor(onSelectPhonemeId) {
    this.onSelectPhonemeId = onSelectPhonemeId;
    this.lyrics = [];
    this.animFrame = null;

    this.canvas = document.getElementById("daisy-canvas");
    this.ctx = this.canvas.getContext("2d");
    this.ribbonEl = document.getElementById("daisy-lyrics-ribbon");
    this.audioEl = document.getElementById("daisy-audio-el");
    this.btnPlay = document.getElementById("btn-play-daisy");
    this.hudAlias = document.getElementById("hud-alias");
    this.hudPitch = document.getElementById("hud-pitch");

    this._bindEvents();
  }

  loadTrack(audioUrl, timedLyrics) {
    this.lyrics = timedLyrics || [];
    this.audioEl.src = audioUrl;
    this._renderLyricsRibbon();
    this._drawPianoRoll(0);
  }

  _bindEvents() {
    this.btnPlay.addEventListener("click", () => {
      if (this.audioEl.paused) {
        this.audioEl.play();
      } else {
        this.audioEl.pause();
      }
    });

    this.audioEl.addEventListener("play", () => {
      this.btnPlay.innerHTML = `&#10074;&#10074; Pause Daisy Bell`;
      this._startSyncLoop();
    });

    this.audioEl.addEventListener("pause", () => {
      this.btnPlay.innerHTML = `&#9654; Play Daisy Bell`;
      cancelAnimationFrame(this.animFrame);
      this._updateSyncState(this.audioEl.currentTime);
    });

    this.audioEl.addEventListener("ended", () => {
      this.btnPlay.innerHTML = `&#9654; Play Daisy Bell`;
      cancelAnimationFrame(this.animFrame);
      this._updateSyncState(0);
    });

    this.audioEl.addEventListener("timeupdate", () => {
      if (this.audioEl.paused) {
        this._updateSyncState(this.audioEl.currentTime);
      }
    });
  }

  _renderLyricsRibbon() {
    this.ribbonEl.innerHTML = this.lyrics
      .map(
        (item, idx) => `
      <button type="button" class="lyric-token" data-idx="${idx}" title="Click to jump to ${item.lyric} (${item.alias})">
        <strong>${item.lyric}</strong>
        <small>[${item.alias}]</small>
      </button>
    `
      )
      .join("");

    this.ribbonEl.querySelectorAll(".lyric-token").forEach((btn) => {
      btn.addEventListener("click", () => {
        const idx = Number(btn.dataset.idx);
        const item = this.lyrics[idx];
        if (!item) return;
        this.audioEl.currentTime = item.start_sec;
        this.audioEl.play();
        if (this.onSelectPhonemeId && item.sample_id) {
          this.onSelectPhonemeId(item.sample_id, false);
        }
      });
    });
  }

  _startSyncLoop() {
    const tick = () => {
      this._updateSyncState(this.audioEl.currentTime);
      if (!this.audioEl.paused && !this.audioEl.ended) {
        this.animFrame = requestAnimationFrame(tick);
      }
    };
    this.animFrame = requestAnimationFrame(tick);
  }

  _updateSyncState(currentTime) {
    let activeNote = null;
    const buttons = this.ribbonEl.querySelectorAll(".lyric-token");

    this.lyrics.forEach((item, idx) => {
      const btn = buttons[idx];
      if (!btn) return;
      btn.classList.remove("active", "sung");
      if (currentTime >= item.start_sec && currentTime <= item.end_sec) {
        btn.classList.add("active");
        activeNote = item;
      } else if (currentTime > item.end_sec) {
        btn.classList.add("sung");
      }
    });

    if (activeNote) {
      this.hudAlias.textContent = `[ ${activeNote.alias} ]`;
      this.hudPitch.textContent = `${midiToNoteName(activeNote.midi)} (${activeNote.freq_hz} Hz)`;
    } else if (currentTime === 0) {
      this.hudAlias.textContent = `[ d ey ]`;
      this.hudPitch.textContent = `C5 (523.3 Hz)`;
    }

    this._drawPianoRoll(currentTime);
  }

  _drawPianoRoll(currentTime) {
    const w = this.canvas.width;
    const h = this.canvas.height;
    const ctx = this.ctx;

    ctx.fillStyle = "#060911";
    ctx.fillRect(0, 0, w, h);

    if (!this.lyrics.length) return;

    const totalSec = Math.max(21.5, this.lyrics[this.lyrics.length - 1].end_sec + 0.8);
    const midis = this.lyrics.map((n) => n.midi);
    const minMidi = Math.min(...midis) - 2;
    const maxMidi = Math.max(...midis) + 2;

    // Horizontal pitch grid lines
    ctx.strokeStyle = "rgba(38, 51, 77, 0.45)";
    ctx.lineWidth = 1;
    for (let m = minMidi; m <= maxMidi; m += 2) {
      const y = h - ((m - minMidi) / (maxMidi - minMidi)) * (h - 24) - 12;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // Draw note blocks
    for (const n of this.lyrics) {
      const x = (n.start_sec / totalSec) * w;
      const nw = Math.max(8, ((n.end_sec - n.start_sec) / totalSec) * w - 2);
      const y = h - ((n.midi - minMidi) / (maxMidi - minMidi)) * (h - 32) - 20;
      const nh = 14;

      const isActive = currentTime >= n.start_sec && currentTime <= n.end_sec;
      const isPast = currentTime > n.end_sec;

      ctx.fillStyle = isActive
        ? "#38bdf8"
        : isPast
        ? "rgba(16, 185, 129, 0.72)"
        : "rgba(99, 102, 241, 0.65)";
      ctx.beginPath();
      ctx.roundRect(x, y, nw, nh, 4);
      ctx.fill();

      if (isActive) {
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      ctx.fillStyle = "#f8fafc";
      ctx.font = "600 10px Inter, sans-serif";
      ctx.fillText(n.lyric, x + 4, y - 3);
    }

    // Playhead cursor
    const playX = (currentTime / totalSec) * w;
    ctx.strokeStyle = "#fcd34d";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(playX, 0);
    ctx.lineTo(playX, h);
    ctx.stroke();
  }
}
