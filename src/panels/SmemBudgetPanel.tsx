import {
  blkMMult,
  blkNMult,
  inst,
  kStages,
  summary,
  tileK,
  world,
} from '../state';
import { buildBudget, kIterations, maxStages, type Segment } from '../smem_budget';
import { TruthFooter } from './TruthFooter';

const COLOR: Record<Segment['kind'], string> = {
  A: '#7ec699',
  B: '#7e9cd8',
  pad: '#3a3a3a',
  mbar: '#b48ead',
  unused: '#1a1f2a',
};

// Phase 6 — variant-specific auxiliary segment colours (visual only, not
// added to the Segment type since math modules are off-limits).
const META_COLOR = '#b48ead';   // --mbar purple for 2:4 metadata
const SCALE_COLOR = '#e0cf7a';  // --warn yellow for SFA/SFB

export function SmemBudgetPanel() {
  const i = inst.value;
  const stages = kStages.value;
  const tileKValue = tileK.value;
  const bm = blkMMult.value;
  const bn = blkNMult.value;
  const mult = { blkMMult: bm, blkNMult: bn };
  const budget = buildBudget(i, stages, mult);
  const iters = kIterations(i, tileKValue);
  const maxS = maxStages(i, 0, mult);
  // Phase 6 — variant extras. Approximate auxiliary bytes per stage from the
  // simulator summary, which knows linesPerSlab_metadata and linesPerSlab_scale
  // (128 B lines). We don't modify smem_budget.ts; this is visual-only.
  const sumV = summary.value;
  const metaBytesPerStage = sumV.extras.sparse ? sumV.linesPerSlab_metadata * 128 : 0;
  const scaleBytesPerStage = sumV.extras.blockScaled ? sumV.linesPerSlab_scale * 128 : 0;

  // Active stage — the ring slot currently being filled (producer) or read
  // (consumer). Single source of truth: world.producerTransfer?.stage ??
  // world.consumerAtom?.stage (Phase 4 wires the latter).
  const wNow = world.value;
  const pt = wNow.producerTransfer;
  const activeStageConsumer: number | null = pt && pt.stage >= 0 ? pt.stage : null;

  // Per-stage fillFrac for the zoom bar — visualises stage "filling" during
  // a TMA load. 1 while 'hold'/'consume', ramps 0→1 while 'fill', drops to 0
  // on 'empty'. See plan §B (ring.fillFrac) and §D6.
  const ringFillFrac: number[] = Array.from({ length: stages }, (_, s) =>
    wNow.ring[s]?.fillFrac ?? 0,
  );

  return (
    <div class="panel smem-budget">
      <h3>
        SMEM budget{' '}
        <small>
          — {(budget.total / 1024).toFixed(0)} KiB/SM · <code>kStages={stages}</code> ·{' '}
          <code>BLK_M×BLK_K = {i.M * bm}×{i.K}</code>, <code>BLK_N×BLK_K = {i.N * bn}×{i.K}</code> ·{' '}
          {(budget.usedBytes / 1024).toFixed(1)} KiB used{' '}
          {budget.fits ? (
            <span class="budget__fits">fits</span>
          ) : (
            <span class="budget__overflow">OVERFLOW — max {maxS} stages</span>
          )}
        </small>
      </h3>

      <div class="budget__bar" title="relative to SMEM capacity">
        {budget.segments.map((seg, idx) => {
          const pct = (seg.bytes / budget.total) * 100;
          const isActive =
            activeStageConsumer !== null &&
            (seg.kind === 'A' || seg.kind === 'B') &&
            seg.stage === activeStageConsumer;
          return (
            <div
              key={`${seg.kind}.${seg.stage ?? 0}.${idx}`}
              class={`budget__seg budget__seg--${seg.kind} ${isActive ? 'is-active' : ''}`}
              style={{ width: `${pct}%`, background: COLOR[seg.kind] }}
              title={`${seg.label} · ${seg.bytes} B (${pct.toFixed(2)}%)`}
            >
              <span class="budget__seg-label">{seg.label}</span>
            </div>
          );
        })}
      </div>

      {/* Zoomed per-stage cost — ignores the unused part so the A/B/pad/mbar
          segments are legibly sized regardless of kStages × blkMult. */}
      <div class="budget__zoombar-head">
        <span class="budget__zoombar-lbl">per-stage cost (zoomed)</span>
        <code class="budget__zoombar-stage">
          stage_bytes = {budget.stageBytes.stage} B
        </code>
      </div>
      <div class="budget__zoombar">
        {Array.from({ length: stages }, (_, s) => {
          const sb = budget.stageBytes;
          // Phase 6: additional variant segments per stage (sparse meta,
          // block_scaled SFA/SFB). Tacked onto the end of the stage bar.
          type ZSegKind = 'A' | 'B' | 'pad' | 'mbar' | 'meta' | 'scaleA' | 'scaleB';
          const parts: { kind: ZSegKind; bytes: number }[] = [
            { kind: 'A', bytes: sb.a },
            { kind: 'B', bytes: sb.b },
            ...(sb.pad > 0 ? [{ kind: 'pad' as const, bytes: sb.pad }] : []),
            { kind: 'mbar' as const, bytes: sb.mbar },
          ];
          if (metaBytesPerStage > 0) {
            parts.push({ kind: 'meta', bytes: metaBytesPerStage });
          }
          if (scaleBytesPerStage > 0) {
            // Split into SFA + SFB halves (~equal) for visual distinction.
            const half = Math.max(1, Math.floor(scaleBytesPerStage / 2));
            parts.push({ kind: 'scaleA', bytes: half });
            parts.push({ kind: 'scaleB', bytes: scaleBytesPerStage - half });
          }
          // Total visual budget includes the auxiliaries for sizing.
          const stageVisualTotal =
            sb.stage + metaBytesPerStage + scaleBytesPerStage;
          const isActiveStage = activeStageConsumer === s;
          const fillFrac = ringFillFrac[s] ?? 0;
          return (
            <div
              key={`zoom-${s}`}
              class={`budget__zoomstage ${isActiveStage ? 'is-active' : ''}`}
              title={`stage ${s} · ${sb.stage} B total · fill ${(fillFrac * 100).toFixed(0)}%`}
            >
              <span class="budget__zoomstage-lbl">s{s}</span>
              <div class="budget__zoomstage-bar">
                {parts.map((p, pi) => {
                  const segBg =
                    p.kind === 'meta'
                      ? `repeating-linear-gradient(45deg, ${META_COLOR}, ${META_COLOR} 4px, #6a5470 4px, #6a5470 6px)`
                      : p.kind === 'scaleA' || p.kind === 'scaleB'
                        ? `repeating-linear-gradient(45deg, ${SCALE_COLOR}, ${SCALE_COLOR} 4px, #8a7a3a 4px, #8a7a3a 6px)`
                        : undefined;
                  return (
                  <span
                    key={`${s}-${pi}`}
                    class={`budget__zoomseg budget__zoomseg--${p.kind}`}
                    style={{
                      flexBasis: `${(p.bytes / stageVisualTotal) * 100}%`,
                      // fillFrac drives the stage "filling" animation during TMA.
                      // Non-A/B segments (pad, mbar) remain fully visible to
                      // preserve the layout; only A/B fade in with the load.
                      opacity:
                        p.kind === 'A' || p.kind === 'B'
                          ? 0.35 + 0.65 * fillFrac
                          : 1,
                      ...(segBg ? { background: segBg } : {}),
                    }}
                    title={`${p.kind} · ${p.bytes}B · fill ${(fillFrac * 100).toFixed(0)}%`}
                  >
                    {p.bytes >= 256 && (
                      <span class="budget__zoomseg-lbl">{p.kind} {(p.bytes / 1024).toFixed(1)}K</span>
                    )}
                  </span>
                  );
                })}
                {/* Fill overlay: covers the empty portion of the stage during
                    an active TMA load. Hidden when fillFrac===1 (steady hold
                    or consume) so the segmented colours remain readable. */}
                {fillFrac < 1 && (
                  <span
                    class="budget__zoomfill"
                    style={{
                      position: 'absolute',
                      right: 0,
                      top: 0,
                      bottom: 0,
                      width: `${(1 - fillFrac) * 100}%`,
                      background: 'rgba(10, 14, 20, 0.6)',
                      pointerEvents: 'none',
                    }}
                  />
                )}
              </div>
            </div>
          );
        })}
      </div>

      <div class="budget__controls">
        <label>
          kStages
          <select value={stages} onChange={(e) => (kStages.value = Number((e.target as HTMLSelectElement).value))}>
            {[1, 2, 3, 4, 5, 6, 7, 8].map((s) => (
              <option value={s} disabled={s > maxS}>
                {s}
                {s === maxS ? ' (max)' : ''}
              </option>
            ))}
          </select>
        </label>
        <label>
          tileK
          <select value={tileKValue} onChange={(e) => (tileK.value = Number((e.target as HTMLSelectElement).value))}>
            {[1, 2, 4, 8, 16, 32].map((mult) => (
              <option value={i.K * mult}>
                {i.K * mult} ({mult} iter)
              </option>
            ))}
          </select>
        </label>
        <span class="budget__foot">
          stage: A = {budget.stageBytes.a} B, B = {budget.stageBytes.b} B, pad ={' '}
          {budget.stageBytes.pad} B, mbar = {budget.stageBytes.mbar} B → stage_bytes ={' '}
          {budget.stageBytes.stage} B · K iter = {iters} · consumer stage:{' '}
          <code>{activeStageConsumer ?? '—'}</code>
        </span>
      </div>
      <p class="budget__legend">
        <span class="budget__swatch" style={{ background: COLOR.A }} /> A tile
        <span class="budget__swatch" style={{ background: COLOR.B }} /> B tile
        <span class="budget__swatch" style={{ background: COLOR.pad }} /> 128 B pad (A+B aligned as a pair)
        <span class="budget__swatch" style={{ background: COLOR.mbar }} /> mbar (PipelineTmaAsync state)
        {!budget.fits && (
          <span class="budget__legend-fit">
            — try <code>kStages = {Math.max(1, maxS)}</code>
          </span>
        )}
      </p>
      <TruthFooter
        models="CUTLASS stage byte budget (A + B + pad + mbar) from smem_budget.ts; active-stage highlight tracks world.producerTransfer.stage ?? world.consumerAtom.stage; per-stage A/B segment opacity animates with world.ring.fillFrac during the active TMA; metadata/SFA/SFB segments included for sparse / block_scaled variants."
        schematic="metadata and scale tensor sizes are line-granular estimates; dynamic SMEM carveout not modelled; real epilogue-staging bytes are not counted."
        cite="smem_budget.ts · buildBudget; gemm/collective/builders/sm90_gmma_builder.inl"
      />
    </div>
  );
}
