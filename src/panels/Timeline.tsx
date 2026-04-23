import { useEffect, useMemo } from 'preact/hooks';
import {
  consumerPhases,
  currentConsumerPhase,
  currentEpiloguePhase,
  currentProducerPhase,
  epiloguePhases,
  pipelineMode,
  playbackRate,
  playing,
  producerPhases,
  summary,
  tick,
  totalTicks,
  kStages,
  inst,
  world,
} from '../state';
import { pipelineAnnotations, ringState, type StallSpan } from '../pipeline_state';
import { TruthFooter } from './TruthFooter';
import type { Phase } from '../state';

// Swim lane assignment. Under warp-specialized, producer (TMA on one warpgroup)
// and consumer (MMA on another) really do run concurrently; under coupled,
// producer+consumer are legs of the same warp and collapse visually.
type Lane = 'producer' | 'consumer' | 'epilogue';

const LANE_ORDER: Lane[] = ['producer', 'consumer', 'epilogue'];
const LANE_LABEL: Record<Lane, string> = {
  producer: 'producer (TMA / cp.async)',
  consumer: 'consumer (MMA step)',
  epilogue: 'epilogue (acc → GMEM)',
};

const LANE_H = 28;
const RING_H = 18;
const MBAR_H = 22;
const GAP = 4;
const LEFT = 132;
const PAD_R = 12;

export function Timeline() {
  const T = totalTicks.value;
  const t = tick.value;
  const prodPh = producerPhases.value;
  const consPh = consumerPhases.value;
  const epiPh = epiloguePhases.value;
  const curCons = currentConsumerPhase.value;
  const curProd = currentProducerPhase.value;
  const curEpi = currentEpiloguePhase.value;
  const mode = pipelineMode.value;
  const isPlaying = playing.value;
  const rate = playbackRate.value;
  const stages = kStages.value;
  const i = inst.value;

  // "Highlighted" phase = whichever is primarily driving animations right now.
  const cur: Phase | null = curCons ?? curEpi ?? curProd;

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

  // Derive pipeline annotations from the emitted streams. We surface:
  //   consumer-wait-fill — leading stall before iter 0 fires
  //   producer-wait-empty — producer idle at the tail while consumer drains
  //   mbarrier-arrive / mbarrier-wait — one triangle per event
  const annotations = useMemo<StallSpan[]>(() => {
    if (prodPh.length === 0 || consPh.length === 0) return [];
    // numIters here reflects how many pipelined producer/consumer pairs exist;
    // we use the emitted producer stream length (which tracks the cap) so
    // annotations align with what Timeline actually draws.
    const numIters = prodPh.length;
    return pipelineAnnotations({
      numIters,
      kStages: stages,
      producerStartTick: prodPh[0].startTick,
      producerPhaseTicks: prodPh[0].endTick - prodPh[0].startTick,
      firstConsumerTick: consPh[0].startTick,
      consumerPhaseTicks: consPh[0].endTick - consPh[0].startTick,
    });
  }, [prodPh, consPh, stages, i.id]);

  // Ring occupancy at the current tick (based on consumer iter).
  const ring = useMemo(() => {
    const firstConsumer = consPh[0]?.startTick ?? 0;
    const consumerDur = consPh[0] ? consPh[0].endTick - consPh[0].startTick : 1;
    const iter = Math.max(0, Math.floor((t - firstConsumer) / consumerDur));
    return ringState(Math.max(0, iter), stages);
  }, [t, consPh, stages]);

  // Per-stage mbarrier state over time. We render kStages rows; each cell is
  // green when that stage is "full" (producer arrived, consumer yet to drain)
  // and red when "empty" (consumer drained, producer yet to refill). Derived
  // from the emitted timeline — no heuristics.
  type MbarState = 'empty' | 'full';
  const mbarTimeline = useMemo(() => {
    const rows: { stage: number; spans: { start: number; end: number; state: MbarState }[] }[] = [];
    for (let s = 0; s < stages; s++) {
      const spans: { start: number; end: number; state: MbarState }[] = [];
      // Find all producer arrivals at this stage in order.
      const arrivals = prodPh.filter((ph) => ph.stage === s).map((ph) => ph.endTick);
      const waits = consPh.filter((ph) => ph.stage === s).map((ph) => ph.startTick);
      // Interleave: initial state is "empty". At each arrival → "full"; at each
      // wait → "empty". We emit a span for every interval between events.
      const events: { time: number; kind: 'arrive' | 'wait' }[] = [
        ...arrivals.map((time) => ({ time, kind: 'arrive' as const })),
        ...waits.map((time) => ({ time, kind: 'wait' as const })),
      ].sort((a, b) => a.time - b.time);
      let cursor = 0;
      let state: MbarState = 'empty';
      for (const e of events) {
        if (e.time > cursor) spans.push({ start: cursor, end: e.time, state });
        cursor = e.time;
        state = e.kind === 'arrive' ? 'full' : 'empty';
      }
      if (cursor < T) spans.push({ start: cursor, end: T, state });
      rows.push({ stage: s, spans });
    }
    return rows;
  }, [prodPh, consPh, stages, T]);

  const xPct = (ticks: number) => `${(ticks / T) * 100}%`;
  const wPct = (a: number, b: number) => `${((b - a) / T) * 100}%`;

  // Under coupled regime there are no mbarriers (same warp, no ring) — hide
  // the per-stage strip. Under warpspec, draw one row per stage.
  const showMbar = mode === 'warpspec';
  const mbarH = showMbar ? MBAR_H * stages : 0;
  const showRing = mode === 'warpspec';
  const ringH = showRing ? RING_H : 0;
  const ringGap = showRing ? GAP : 0;
  const totalH = ringH + ringGap + LANE_ORDER.length * (LANE_H + GAP) + mbarH + (showMbar ? GAP : 0);

  const phasesByLane: Record<Lane, Phase[]> = {
    producer: prodPh,
    consumer: consPh,
    epilogue: epiPh,
  };

  const modeLabel = mode === 'warpspec' ? 'ASYNC · producer ∥ consumer' : 'SYNC · same warp';
  const modeTitle =
    mode === 'warpspec'
      ? 'Warp-specialized: producer warpgroup issues tma.load while consumer warpgroup executes wgmma/tcgen05.mma in parallel, coordinated by mbarriers.'
      : 'Coupled: a single warp issues cp.async then mma.sync in sequence. No ring, no mbarrier.';

  return (
    <div class="timeline timeline--lanes">
      <div class="timeline__head">
        <button onClick={() => (playing.value = !isPlaying)} class="timeline__play">
          {isPlaying ? '⏸' : '▶'}
        </button>
        <button onClick={() => (tick.value = Math.max(0, t - 1))} title="−1 tick (←)">◀</button>
        <input
          type="range"
          class="timeline__scrub"
          min={0}
          max={Math.max(0, T - 0.001)}
          step={0.01}
          value={t}
          onInput={(e) => (tick.value = Number((e.target as HTMLInputElement).value))}
        />
        <button onClick={() => (tick.value = Math.min(T - 0.001, t + 1))} title="+1 tick (→)">▶</button>
        <select value={rate} onChange={(e) => (playbackRate.value = Number((e.target as HTMLSelectElement).value))}>
          <option value={0.5}>0.5×</option>
          <option value={1}>1×</option>
          <option value={2}>2×</option>
          <option value={4}>4×</option>
          <option value={8}>8×</option>
        </select>
        <code class="timeline__tick">t {t.toFixed(2)} / {(T - 1).toFixed(0)}</code>
        <span class={`timeline__mode timeline__mode--${mode}`} title={modeTitle}>{modeLabel}</span>
        <span class="timeline__label">{cur?.label ?? '—'}</span>
      </div>

      <div class="timeline__grid" style={{ height: `${totalH}px` }}>
        {/* ring occupancy strip — hidden under coupled (no ring buffer). */}
        {showRing && (
          <div class="timeline__row timeline__row--ring" style={{ top: 0, height: `${RING_H}px` }}>
            <span class="timeline__rowlabel">ring (k={stages})</span>
            <div class="timeline__track timeline__track--ring">
              {ring.map((r, si) => (
                <div
                  key={si}
                  class={`timeline__ringslot timeline__ringslot--${r.role}`}
                  style={{ left: `${(si / stages) * 100}%`, width: `${100 / stages}%` }}
                  title={`stage ${r.stage} · slice ${r.slice} · ${r.role}`}
                >
                  <span class="timeline__ringslot-lbl">s{r.stage}:k{r.slice}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* swim lanes — each lane draws its own phase stream; under warpspec
            they overlap in time, under coupled they are strictly serial. */}
        {LANE_ORDER.map((lane, li) => {
          const top = ringH + ringGap + li * (LANE_H + GAP);
          const lanePhases = phasesByLane[lane];
          return (
            <div class="timeline__row" style={{ top: `${top}px`, height: `${LANE_H}px` }}>
              <span class="timeline__rowlabel">{LANE_LABEL[lane]}</span>
              <div class="timeline__track">
                {/* stall shading on this lane */}
                {annotations
                  .filter((a) => (
                    (lane === 'consumer' && a.kind === 'consumer-wait-fill') ||
                    (lane === 'producer' && a.kind === 'producer-wait-empty')
                  ))
                  .map((a, ai) => (
                    <div
                      key={`stall-${ai}`}
                      class={`timeline__stall timeline__stall--${a.kind}`}
                      style={{ left: xPct(a.startTick), width: wPct(a.startTick, a.endTick) }}
                      title={a.label}
                    />
                  ))}
                {/* phases */}
                {lanePhases.map((ph) => {
                  // Sub-atom micro-strip: only drawn on consumer mma-step
                  // phases when atomsPerStage_MN > 1. Shows a thin row of
                  // tick marks at positions i/atomsPerStage_MN inside the
                  // bar; the current MN atom (from world.consumerAtom) lights
                  // up when this bar is the active phase.
                  const atomsMN = summary.value.atomsPerStage_MN;
                  const isMmaStep =
                    lane === 'consumer' &&
                    (ph.kind === 'wgmma.step' ||
                      ph.kind === 'tcgen05.mma.step') &&
                    !ph.collapsedCount;
                  const showStrip = isMmaStep && atomsMN > 1;
                  const isCurrent = cur?.id === ph.id;
                  const activeSubIdx =
                    isCurrent && world.value.consumerAtom
                      ? world.value.consumerAtom.atomFlatIdx
                      : -1;
                  return (
                    <div
                      key={ph.id}
                      class={`timeline__phase timeline__phase--${lane} ${isCurrent ? 'is-active' : ''}`}
                      style={{ left: xPct(ph.startTick), width: wPct(ph.startTick, ph.endTick) }}
                      title={`${ph.label}\n${ph.description}`}
                      onClick={() => (tick.value = ph.startTick)}
                    >
                      <span class="timeline__phase-lbl">{ph.label}</span>
                      {showStrip && (
                        <span class="timeline__subatom-strip">
                          {Array.from({ length: atomsMN }, (_, i) => (
                            <span
                              key={`sa-${i}`}
                              class={`timeline__subatom-tick ${activeSubIdx === i ? 'is-active' : ''}`}
                              style={{ left: `${(i / atomsMN) * 100}%`, width: `${100 / atomsMN}%` }}
                              title={`MN atom ${i} of ${atomsMN}`}
                            />
                          ))}
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}

        {/* mbarrier strip — one row per stage showing full (green) / empty
            (dim) over time. Arrive triangles on producer→full transitions,
            wait triangles on consumer→empty transitions. Hidden under the
            coupled regime because that path uses no mbarriers. */}
        {showMbar && mbarTimeline.map((row, ri) => (
          <div
            key={`mbar-${row.stage}`}
            class="timeline__row timeline__row--mbar"
            style={{
              top: `${ringH + ringGap + LANE_ORDER.length * (LANE_H + GAP) + ri * MBAR_H}px`,
              height: `${MBAR_H}px`,
            }}
          >
            <span class="timeline__rowlabel">mbar[{row.stage}]</span>
            <div class="timeline__track timeline__track--mbar">
              {row.spans.map((span, si) => (
                <div
                  key={`span-${si}`}
                  class={`timeline__mbarspan timeline__mbarspan--${span.state}`}
                  style={{ left: xPct(span.start), width: wPct(span.start, span.end) }}
                  title={`stage ${row.stage} ${span.state} from t=${span.start.toFixed(1)} to t=${span.end.toFixed(1)}`}
                />
              ))}
              {/* event triangles */}
              {annotations
                .filter((a) =>
                  (a.kind === 'mbarrier-arrive' || a.kind === 'mbarrier-wait') &&
                  a.stage === row.stage,
                )
                .map((a, ai) => (
                  <div
                    key={`mb-${ai}`}
                    class={`timeline__mbar timeline__mbar--${a.kind}`}
                    style={{ left: xPct(a.startTick) }}
                    title={a.label}
                  />
                ))}
            </div>
          </div>
        ))}

        {/* global cursor */}
        <div class="timeline__cursor" style={{ left: `calc(${LEFT}px + (100% - ${LEFT + PAD_R}px) * ${t / T})` }} />
      </div>

      {cur && (
        <p class="timeline__desc">
          {curCons && curProd && curCons !== curProd && (
            <span class="timeline__concurrent">
              concurrent: producer · <code>{curProd.label}</code> ∥ consumer · <code>{curCons.label}</code> ·{' '}
            </span>
          )}
          {cur.description}
        </p>
      )}
      <TruthFooter
        models="producer / consumer / epilogue streams with real tick-level overlap (warpspec) or strict serialization (coupled) from simulate(); mbarrier full/empty state derived from arrive/wait events in world.mbar; ring occupancy with per-stage fillFrac follows the consumer tick cursor; sub-atom micro-strip reflects the MN atom sweep within each consumer phase."
        schematic="TMA / MMA cycle counts are nominal per-inst defaults (not hardware-measured); real mbarrier transactions have a small latency not modelled; collapse bars (× N more) use an averaged sub-atom pointer; pattern-specific lane-wave maxWays is derived from ldmatrix.x4.N access pattern only."
        cite="simulation.ts · simulate; pipeline_state.ts · emitTimeline; cutlass/pipeline/sm90_pipeline.hpp · PipelineTmaAsync"
      />
    </div>
  );
}
