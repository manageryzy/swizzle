import { render } from 'preact';
import { effect } from '@preact/signals';
import { App } from './App';
import { hydrateFromUrl, installUrlSync } from './url_state';
import {
  inst,
  resetTickIfInstChanged,
  tick,
  playing,
  phases,
  totalTicks,
  densityMode,
  laneSel,
  warpSel,
  warpsInGroup,
  cycleSel,
  currentConflicts,
} from './state';

hydrateFromUrl();
installUrlSync();

// Reset tick on instruction switch so we don't scrub into a dead phase.
effect(() => {
  resetTickIfInstChanged(inst.value.id);
});

// Keyboard shortcuts. Registered once at startup; ignored while the user is
// typing in an <input> / <select> / <textarea> so the ConfigBar dropdowns
// still work with arrow keys.
function isTypingTarget(el: EventTarget | null): boolean {
  if (!(el instanceof HTMLElement)) return false;
  const tag = el.tagName;
  if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return true;
  if (el.isContentEditable) return true;
  return false;
}

window.addEventListener('keydown', (e) => {
  if (isTypingTarget(e.target)) return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;

  switch (e.key) {
    case ' ':
      e.preventDefault();
      playing.value = !playing.value;
      return;
    case 'ArrowLeft':
      e.preventDefault();
      if (e.shiftKey) {
        // Jump to previous phase start.
        const t = tick.value;
        const prev = [...phases.value].reverse().find((p) => p.startTick < t);
        tick.value = prev ? prev.startTick : 0;
      } else {
        tick.value = Math.max(0, tick.value - 1);
      }
      return;
    case 'ArrowRight':
      e.preventDefault();
      if (e.shiftKey) {
        const t = tick.value;
        const next = phases.value.find((p) => p.startTick > t);
        tick.value = next ? next.startTick : Math.max(0, totalTicks.value - 1);
      } else {
        tick.value = Math.min(totalTicks.value - 0.001, tick.value + 1);
      }
      return;
    case '[':
      e.preventDefault();
      cycleSel.value = Math.max(0, cycleSel.value - 1);
      return;
    case ']': {
      e.preventDefault();
      const maxCycle = Math.max(0, (currentConflicts.value[0]?.way ?? 1) - 1);
      cycleSel.value = Math.min(maxCycle, cycleSel.value + 1);
      return;
    }
    case 'l':
    case 'L': {
      e.preventDefault();
      const cur = laneSel.value;
      laneSel.value = cur == null ? 0 : cur >= 31 ? null : cur + 1;
      return;
    }
    case 'w':
    case 'W': {
      e.preventDefault();
      warpSel.value = (warpSel.value + 1) % Math.max(1, warpsInGroup.value);
      return;
    }
    case 'd':
    case 'D':
      e.preventDefault();
      densityMode.value = densityMode.value === 'compact' ? 'detail' : 'compact';
      return;
  }
});

const root = document.getElementById('app');
if (root) render(<App />, root);
