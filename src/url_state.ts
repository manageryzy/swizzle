// URL hash state (e.g. #inst=sm100.tcgen05.cg1.m128n128k16.f16&swizzle=128B&t=5).
// Hash beats query string here because GitHub Pages preserves it across the
// static hosting without server routing.

import { effect } from '@preact/signals';
import { spec, tick, activePatternId, playing } from './state';
import { SWIZZLES, type SwizzleKind } from './swizzle';
import { INSTRUCTIONS } from './instructions';
import { PATTERNS } from './patterns';

const KEYS = ['inst', 'swizzle', 'ma', 'mb', 'src', 'pat', 't'] as const;
type Key = (typeof KEYS)[number];

function read(): Partial<Record<Key, string>> {
  const h = window.location.hash.replace(/^#/, '');
  if (!h) return {};
  const out: Partial<Record<Key, string>> = {};
  for (const part of h.split('&')) {
    const [k, v] = part.split('=');
    if (k && v && (KEYS as readonly string[]).includes(k)) {
      out[k as Key] = decodeURIComponent(v);
    }
  }
  return out;
}

function write(state: Record<Key, string>) {
  const kept = Object.entries(state).filter(([, v]) => v !== '');
  const h = kept.map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join('&');
  const next = h ? `#${h}` : '';
  if (window.location.hash !== next) {
    // Use history.replaceState so the back button doesn't fill with ticks.
    history.replaceState(null, '', `${window.location.pathname}${window.location.search}${next}`);
  }
}

export function hydrateFromUrl() {
  const p = read();
  const s = spec.value;
  let nextSpec = { ...s };

  if (p.inst && INSTRUCTIONS.some((i) => i.id === p.inst)) nextSpec.instId = p.inst;
  if (p.swizzle && p.swizzle in SWIZZLES) nextSpec.swizzle = p.swizzle as SwizzleKind;
  if (p.ma === 'K' || p.ma === 'MN') nextSpec.majorA = p.ma;
  if (p.mb === 'K' || p.mb === 'MN') nextSpec.majorB = p.mb;
  if (p.src === 'smem' || p.src === 'rmem' || p.src === 'tmem') nextSpec.aSource = p.src;
  spec.value = nextSpec;

  if (p.pat && p.pat in PATTERNS) activePatternId.value = p.pat;
  if (p.t !== undefined) {
    const n = Number(p.t);
    if (Number.isFinite(n)) tick.value = n;
  }
}

let writing = false;

export function installUrlSync() {
  effect(() => {
    // Short-circuit while playing so we don't hit `history.replaceState` at
    // ~60 Hz. Reading `tick.value` only when paused also unsubscribes this
    // effect from tick during playback, so it runs exactly once on each
    // playing→paused edge (and once on every tick when paused).
    const s = spec.value;
    const pat = activePatternId.value;
    const isPlaying = playing.value;
    if (writing || isPlaying) return;
    const t = tick.value;
    writing = true;
    try {
      write({
        inst: s.instId,
        swizzle: s.swizzle,
        ma: s.majorA,
        mb: s.majorB,
        src: s.aSource,
        pat,
        t: t.toFixed(2),
      });
    } finally {
      writing = false;
    }
  });

  window.addEventListener('hashchange', () => {
    writing = true;
    try {
      hydrateFromUrl();
    } finally {
      writing = false;
    }
  });
}
