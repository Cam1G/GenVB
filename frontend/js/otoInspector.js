/**
 * Interactive OpenUTAU oto.ini Phoneme Waveform & Timing Boundary Inspector.
 */

export class OtoInspectorController {
  constructor() {
    this.samples = [];
    this.filteredSamples = [];
    this.selectedSample = null;
    this.activeFilter = "all";
    this.searchQuery = "";
    this.sampleAudio = new Audio();

    this.listEl = document.getElementById("sample-list");
    this.searchEl = document.getElementById("phoneme-search");
    this.pillsEl = document.getElementById("phoneme-filter-pills");
    this.canvas = document.getElementById("oto-canvas");
    this.ctx = this.canvas.getContext("2d");

    this.titleEl = document.getElementById("oto-selected-title");
    this.descEl = document.getElementById("oto-selected-desc");
    this.btnPlay = document.getElementById("btn-play-sample");
    this.valOffset = document.getElementById("val-offset");
    this.valOverlap = document.getElementById("val-overlap");
    this.valPreutt = document.getElementById("val-preutt");
    this.valConsonant = document.getElementById("val-consonant");
    this.valCutoff = document.getElementById("val-cutoff");
    this.rawLineEl = document.getElementById("oto-raw-line");
    this.aliasesBox = document.getElementById("oto-aliases-list");

    this._bindEvents();
  }

  setSamples(samples) {
    this.samples = samples || [];
    this._applyFilter();
    if (this.samples.length > 0) {
      // Select "16_d_ey" ("Dai-") by default so user immediately sees a rich CV phoneme
      const defaultItem = this.samples.find((s) => s.id === "16_d_ey") || this.samples[0];
      this.selectSampleById(defaultItem.id, false);
    }
  }

  selectSampleById(sampleId, autoPlay = true) {
    const found = this.samples.find((s) => s.id === sampleId);
    if (!found) return;
    this.selectedSample = found;

    this.listEl.querySelectorAll(".sample-item").forEach((el) => {
      el.classList.toggle("active", el.dataset.id === sampleId);
    });

    this.titleEl.textContent = `${found.filename} — "${found.word}" [${found.arpabet}]`;
    this.descEl.textContent = found.description;

    const o = found.oto;
    this.valOffset.textContent = `${o.offset} ms`;
    this.valOverlap.textContent = `${o.overlap} ms`;
    this.valPreutt.textContent = `${o.preutterance} ms`;
    this.valConsonant.textContent = `${o.consonant} ms`;
    this.valCutoff.textContent = `${o.cutoff} ms`;

    const primaryAlias = found.aliases[0] || found.arpabet;
    this.rawLineEl.textContent = `${found.filename}=${primaryAlias},${o.offset},${o.consonant},${o.cutoff},${o.preutterance},${o.overlap}`;

    this.aliasesBox.innerHTML = found.aliases
      .map((a) => `<span class="alias-chip">${a}</span>`)
      .join("");

    this._drawOtoWaveform(found);

    if (autoPlay && found.audio_url) {
      this.playSelectedSample();
    }
  }

  playSelectedSample() {
    if (!this.selectedSample?.audio_url) return;
    this.sampleAudio.src = this.selectedSample.audio_url;
    this.sampleAudio.currentTime = 0;
    this.sampleAudio.play();
  }

  _bindEvents() {
    this.searchEl.addEventListener("input", (e) => {
      this.searchQuery = e.target.value.trim().toLowerCase();
      this._applyFilter();
    });

    this.pillsEl.addEventListener("click", (e) => {
      const btn = e.target.closest(".pill-btn");
      if (!btn) return;
      this.pillsEl.querySelectorAll(".pill-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      this.activeFilter = btn.dataset.filter;
      this._applyFilter();
    });

    this.btnPlay.addEventListener("click", () => {
      this.playSelectedSample();
    });
  }

  _applyFilter() {
    this.filteredSamples = this.samples.filter((s) => {
      if (this.activeFilter !== "all" && s.type !== this.activeFilter) {
        return false;
      }
      if (!this.searchQuery) return true;
      const hay = `${s.id} ${s.word} ${s.arpabet} ${s.aliases.join(" ")}`.toLowerCase();
      return hay.includes(this.searchQuery);
    });

    this.listEl.innerHTML = this.filteredSamples
      .map(
        (s) => `
      <button type="button" class="sample-item ${
        this.selectedSample?.id === s.id ? "active" : ""
      }" data-id="${s.id}">
        <div class="sample-item-left">
          <strong>${s.filename}</strong>
          <span>"${s.word}" &bull; ${s.aliases.slice(0, 3).join(", ")}</span>
        </div>
        <span class="sample-type-tag">${s.type}</span>
      </button>
    `
      )
      .join("");

    this.listEl.querySelectorAll(".sample-item").forEach((btn) => {
      btn.addEventListener("click", () => {
        this.selectSampleById(btn.dataset.id, true);
      });
    });
  }

  _drawOtoWaveform(sample) {
    const w = this.canvas.width;
    const h = this.canvas.height;
    const ctx = this.ctx;

    ctx.fillStyle = "#05080f";
    ctx.fillRect(0, 0, w, h);

    const o = sample.oto;
    const totalMs = Math.max(250, o.duration_ms || 650);

    const msToX = (ms) => Math.max(0, Math.min(w, (ms / totalMs) * w));

    const offsetX = msToX(o.offset);
    const overlapX = msToX(o.offset + o.overlap);
    const preuttX = msToX(o.offset + o.preutterance);
    const consonantX = msToX(o.offset + o.consonant);
    const cutoffMs = o.cutoff < 0 ? totalMs + o.cutoff : o.offset + o.cutoff;
    const cutoffX = msToX(cutoffMs);

    // 1. Shaded Left Blank (Offset) region
    ctx.fillStyle = "rgba(59, 130, 246, 0.20)";
    ctx.fillRect(0, 0, offsetX, h);

    // 2. Shaded Fixed Consonant (Unstretched) region
    ctx.fillStyle = "rgba(236, 72, 153, 0.16)";
    ctx.fillRect(offsetX, 0, Math.max(0, consonantX - offsetX), h);

    // 3. Shaded Stretchable Vowel region
    ctx.fillStyle = "rgba(56, 189, 248, 0.06)";
    ctx.fillRect(consonantX, 0, Math.max(0, cutoffX - consonantX), h);

    // 4. Shaded Right Blank (Cutoff) region
    ctx.fillStyle = "rgba(245, 158, 11, 0.20)";
    ctx.fillRect(cutoffX, 0, w - cutoffX, h);

    // Center axis
    ctx.strokeStyle = "rgba(148, 163, 184, 0.2)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    // Draw waveform peaks
    const peaks = sample.peaks || [];
    if (peaks.length > 0) {
      const barW = w / peaks.length;
      for (let i = 0; i < peaks.length; i++) {
        const x = i * barW;
        const amp = peaks[i] * (h * 0.41);
        const inActive = x >= offsetX && x <= cutoffX;
        ctx.fillStyle = inActive ? "#38bdf8" : "rgba(100, 116, 139, 0.45)";
        ctx.fillRect(x, h / 2 - amp, Math.max(1.5, barW - 1), amp * 2);
      }
    }

    // Draw vertical UTAU boundary markers + labels
    const drawMarker = (x, color, label, topOffset) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();

      ctx.fillStyle = color;
      ctx.font = "700 11px JetBrains Mono, monospace";
      const labelX = Math.min(w - 65, Math.max(4, x + 5));
      ctx.fillText(label, labelX, topOffset);
    };

    drawMarker(offsetX, "#3b82f6", "OFFSET", 16);
    drawMarker(overlapX, "#10b981", "OVL", 34);
    drawMarker(preuttX, "#ef4444", "PREUTT", 52);
    drawMarker(consonantX, "#ec4899", "FIXED", 70);
    drawMarker(cutoffX, "#f59e0b", "CUTOFF", 16);
  }
}
