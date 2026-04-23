// SMEM panel — slimmed in v2 to render ONLY the currently-consumed stage,
// preceded by a compact ring-chip row that names the other stages and lets
// users jump to the iteration where each becomes the consumer. The pipeline
// story has moved to TileHierarchyPanel (Row 2 K-walk) so we don't need
// three near-identical canvas triplets here anymore.

import {
  blkMMult,
  blkNMult,
  consumerPhases,
  inst,
  kStages,
  spec,
  tick,
  world,
} from '../state';
import { SmemTileView } from './SmemTileView';
import { SmemXfrmView } from './SmemXfrmView';
import { BitfieldPanel } from './BitfieldPanel';
import { canonicalLayout, canonicalT } from '../gmma_layouts';
import { activeSwizzle } from '../state';
import { bytesOf } from '../smem_budget';
import { TruthFooter } from './TruthFooter';

type AnySource = 'smem' | 'rmem' | 'tmem' | 'gmem-wmma';

function sourceOfOperand(
  aSource: readonly AnySource[] | AnySource[],
  operand: 'A' | 'B',
  userSource: string,
  bSource: readonly AnySource[] | AnySource[],
): AnySource {
  if (operand === 'A') {
    if ((aSource as AnySource[]).includes(userSource as AnySource))
      return userSource as AnySource;
    return (aSource as AnySource[])[0];
  }
  const b = bSource as AnySource[];
  return b.includes('smem') ? 'smem' : b[0];
}

function OperandRow({ operand }: { operand: 'A' | 'B' }) {
  const i = inst.value;
  const s = spec.value;
  const stages = kStages.value;
  const w = world.value;
  const bm = blkMMult.value;
  const bn = blkNMult.value;
  const src = sourceOfOperand(i.aSource, operand, s.aSource, i.bSource);
  const inSmem = src === 'smem';

  const major = operand === 'A' ? s.majorA : s.majorB;
  const dtype = operand === 'A' ? i.aDtypes[0] : i.bDtypes[0];
  const elemBytes = bytesOf(dtype);
  const blkRows = (operand === 'A' ? i.M : i.N) * (operand === 'A' ? bm : bn);

  const descArch: 'sm90' | 'sm100' = i.arch === 'sm100' ? 'sm100' : 'sm90';
  const needsDesc =
    i.family === 'wgmma' || i.family === 'tcgen05' || i.family === 'tcgen05.block_scaled';
  const canon = needsDesc && inSmem ? canonicalLayout(descArch, major, s.swizzle) : null;

  // Active stage from the simulator: prefer the consumer (what's being read
  // right now); fall back to producer (what's being filled); default to 0 for
  // warmup ticks. Slice label comes from world.ring[stage].slice so it matches
  // the actual iteration the simulator has placed in that slot (including
  // back-pressure shifts).
  const consumerStage = w.consumerAtom?.stage;
  const producerStage = w.producerTransfer?.stage;
  const activeStageIdx = consumerStage ?? producerStage ?? 0;
  const ringSlot = w.ring[activeStageIdx];
  const activeStage = ringSlot
    ? { stage: activeStageIdx, slice: ringSlot.slice, role: ringSlot.role }
    : { stage: 0, slice: 0, role: 'empty' as const };
  const iter = w.consumerAtom?.kStep ?? w.producerTransfer?.kSlab ?? 0;

  return (
    <div class="smem-operand">
      <div class="smem-operand__head">
        <span class="smem-operand__name">operand {operand}</span>
        <span class="smem-operand__meta">
          BLK_{operand === 'A' ? 'M' : 'N'} × BLK_K = {blkRows} × {i.K} · {dtype}
          {inSmem ? (
            <>
              {' '}· <code class="pill pill--smem">SMEM</code>
              {' '}· <code>kStages = {stages}</code>
              {' '}· consumer iter <code>{iter}</code>
            </>
          ) : (
            <>
              {' '}· <code class={`pill pill--${src}`}>{src.toUpperCase()}</code> (not in SMEM)
            </>
          )}
        </span>
      </div>
      {inSmem ? (
        <>
          <div class={`stage stage--single is-${activeStage.role}`}>
            <div class="stage__badge">
              <span>stage {activeStage.stage} · slice {activeStage.slice} ({activeStage.role})</span>
              <span class="stage__tag">{activeStage.role}</span>
            </div>
            <div class="smem-operand__views smem-operand__views--3act">
              <SmemTileView operand={operand} view="logical" sliceIdx={activeStage.slice} />
              <SmemXfrmView operand={operand} />
              <SmemTileView operand={operand} view="physical" sliceIdx={activeStage.slice} />
            </div>
          </div>
          {canon && (
            <div class="smem-operand__canon">
              <code>
                {canon.swizzle} ∘ {canon.shape} : {canon.stride}
              </code>
              <span> · T = {canonicalT(elemBytes)}</span>
            </div>
          )}
        </>
      ) : (
        <div class="smem-operand__empty">
          operand {operand} lives in <code>{src}</code> — no SMEM footprint for this instruction
        </div>
      )}
    </div>
  );
}

// The ring-chip row replaces the previous side-by-side stage canvases. Each
// chip names a stage's slice and role; clicking jumps the tick to the iter
// at which that stage becomes the consumer (slice = ringState consumeSlice).
function RingChips() {
  const w = world.value;
  const ring = w.ring;
  const consPh = consumerPhases.value;

  const jumpToIter = (slice: number) => {
    // Find first consumer phase whose kSlab matches; that's when that ring
    // slot's current occupant becomes the "consume" role.
    const target =
      consPh.find((p) => p.kSlab === slice) ??
      consPh[Math.min(slice, consPh.length - 1)];
    if (target) tick.value = target.startTick + 0.001;
  };

  if (ring.length === 0) {
    // wmma / coupled paths have no ring — suppress the strip.
    return null;
  }

  return (
    <div class="ring-chips" role="tablist" aria-label="SMEM stages">
      <span class="ring-chips__label">ring</span>
      {ring.map((st) => (
        <button
          key={`chip-${st.stage}`}
          class={`ring-chip ring-chip--${st.role}`}
          onClick={() => jumpToIter(st.slice)}
          title={`jump to iter where stage ${st.stage} consumes slice k${st.slice} · fill ${(st.fillFrac * 100).toFixed(0)}%`}
        >
          <span class="ring-chip__stage">s{st.stage}</span>
          <span class="ring-chip__slice">k{st.slice}</span>
          <span class="ring-chip__role">{st.role}</span>
        </button>
      ))}
    </div>
  );
}

export function SmemPanel() {
  const sw = activeSwizzle.value;
  const stages = kStages.value;
  return (
    <div class="panel panel--wide smem-panel">
      <h3>
        SMEM tiles{' '}
        <small>
          active stage only · logical (pre-swizzle) → swizzle → physical (post-swizzle) ·{' '}
          <code>Swizzle&lt;{sw.B},{sw.M},{sw.S}&gt;</code> · {stages} stages
        </small>
      </h3>
      <RingChips />
      <OperandRow operand="A" />
      <OperandRow operand="B" />
      <div class="legend">
        <span class="legend__item">
          <span class="legend__swatch legend__swatch--consume" /> active stage shown
        </span>
        <span class="legend__item">
          <span class="legend__swatch legend__swatch--fill" /> other stages: see ring chips above & TileHierarchy K-walk
        </span>
        <span class="legend__item">same cell colour traces one matrix element from logical → physical</span>
      </div>

      <div class="smem-panel__bitfield">
        <BitfieldPanel />
      </div>

      <TruthFooter
        verified
        models="active SMEM stage from world.consumerAtom.stage; ring chip row colour by world.ring.role; atom-boundary highlight tracks consumer's active atomM; swizzle byte permutation applied per offset under Swizzle<B,M,S>; sweep progress tied to producer-stream progress during load, epilogue progress during store."
        schematic="actual SMEM descriptor emission; sub-warp ldmatrix.x4 ordering within an atom; per-lane sweep ordering for ldmatrix fan-out; TMA is a collective op so per-lane sweeps are didactic, not hardware-faithful."
        cite="simulation.ts · world.ring; swizzle.ts · apply; cute/swizzle.hpp"
      />
    </div>
  );
}
