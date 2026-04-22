import { useEffect } from 'preact/hooks';
import {
  currentPhase,
  phases,
  playbackRate,
  playing,
  tick,
  totalTicks,
} from '../state';

export function Timeline() {
  const T = totalTicks.value;
  const t = tick.value;
  const p = phases.value;
  const cur = currentPhase.value;
  const isPlaying = playing.value;
  const rate = playbackRate.value;

  // Drive tick advance via requestAnimationFrame while playing.
  useEffect(() => {
    if (!isPlaying) return;
    let raf = 0;
    let last = performance.now();
    const step = (now: number) => {
      const dt = (now - last) / 1000;
      last = now;
      const next = tick.value + dt * rate;
      tick.value = next >= T ? 0 : next;
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [isPlaying, T, rate]);

  return (
    <div class="timeline">
      <div class="timeline__track">
        {p.map((ph) => {
          const left = (ph.startTick / T) * 100;
          const width = ((ph.endTick - ph.startTick) / T) * 100;
          return (
            <div
              key={ph.id}
              class={`timeline__phase ${cur?.id === ph.id ? 'is-active' : ''}`}
              style={{ left: `${left}%`, width: `${width}%` }}
              title={ph.description}
              onClick={() => (tick.value = ph.startTick)}
            >
              <span>{ph.label}</span>
            </div>
          );
        })}
        <div class="timeline__cursor" style={{ left: `${(t / T) * 100}%` }} />
      </div>
      <div class="timeline__controls">
        <button onClick={() => (playing.value = !isPlaying)} class="timeline__play">
          {isPlaying ? '⏸' : '▶'}
        </button>
        <button onClick={() => (tick.value = Math.max(0, t - 1))}>◀</button>
        <input
          type="range"
          min={0}
          max={T - 1}
          step={0.01}
          value={t}
          onInput={(e) => (tick.value = Number((e.target as HTMLInputElement).value))}
        />
        <button onClick={() => (tick.value = Math.min(T - 1, t + 1))}>▶</button>
        <select value={rate} onChange={(e) => (playbackRate.value = Number((e.target as HTMLSelectElement).value))}>
          <option value={0.5}>0.5×</option>
          <option value={1}>1×</option>
          <option value={2}>2×</option>
          <option value={4}>4×</option>
          <option value={8}>8×</option>
        </select>
        <code>
          tick {t.toFixed(2)} / {(T - 1).toFixed(0)}
        </code>
        <span class="timeline__label">{cur?.label ?? '—'}</span>
      </div>
      {cur && <p class="timeline__desc">{cur.description}</p>}
    </div>
  );
}
