/**
 * Script following + phoneme coverage. The coverage map comes from /api/config
 * (each phoneme's candidate source words, mirroring backend local-mode slicing).
 */

const LOOKAHEAD = 6;

export const normalizeWord = (w) => {
  const n = String(w).toLowerCase().replace(/[’`]/g, "'").replace(/[^a-z']/g, "");
  return n === "im" ? "i'm" : n;
};

export const tokenize = (text) => String(text || "").split(/\s+/).map(normalizeWord).filter(Boolean);

/** Splits sentences into display words: [{ sentence, display, norm, index }]. */
export const buildScript = (sentences) => {
  const words = [];
  sentences.forEach((s, sIdx) => {
    s.text.split(/\s+/).filter(Boolean).forEach((display) => {
      words.push({ sentence: sIdx, display, norm: normalizeWord(display), index: words.length });
    });
  });
  return words;
};

/** Greedy forward alignment: returns the script index right after the last matched word. */
export const scriptCursor = (scriptWords, heardTokens) => {
  let pos = 0;
  for (const tok of heardTokens) {
    const end = Math.min(scriptWords.length, pos + LOOKAHEAD);
    for (let i = pos; i < end; i++) {
      if (scriptWords[i].norm === tok) {
        pos = i + 1;
        break;
      }
    }
  }
  return pos;
};

/**
 * Display state per script word, consistent with the coverage count:
 * "heard" (green), "needed" (red: passed/unheard and required for an uncovered sound),
 * "passed" (dim: passed but not required), or "" (not reached yet).
 */
export const classifyWord = (word, heardSet, cursor, coverage) => {
  if (heardSet.has(word.norm)) return "heard";
  if (word.index >= cursor) return "";
  return coverage.missingWords.includes(word.norm) ? "needed" : "passed";
};

/** Returns { covered, total, missingWords } for a set of heard (normalized) words. */
export const computeCoverage = (coverageMap, heardSet, scriptWordSet) => {
  let covered = 0;
  const missing = new Set();
  for (const ph of coverageMap) {
    if (ph.words.some((w) => heardSet.has(w))) {
      covered++;
    } else if (ph.words.length) {
      missing.add(ph.words.find((w) => scriptWordSet.has(w)) || ph.words[0]);
    }
  }
  return { covered, total: coverageMap.length, missingWords: [...missing] };
};
