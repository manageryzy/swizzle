import { useEffect, useRef, useState } from 'preact/hooks';
import {
  currentConsumerPhase,
  currentEpiloguePhase,
  inst,
  phaseProgress,
  summary,
  world,
} from '../state';
import {
  TMEM_DP,
  TMEM_COLS,
  TMEM_SUBPARTITIONS,
  TCGEN05_SHAPES,
  accFootprint,
  findShape,
} from '../tmem';
import { PTX_LAYOUTS, classifyLayout, peerMRange } from '../tcgen05_layouts';
import { TruthFooter } from './TruthFooter';

const CELL = 2;           // per-col pixel height of a DP row is CELL+1
const GUTTER = 6;         // gap between subpartition blocks
const DP_PER_SUB = TMEM_DP / TMEM_SUBPARTITIONS; // 32

function subColor(sub: number): string {
  return `hsl(${(sub * 360) / TMEM_SUBPARTITIONS}, 55%, 35%)`;
}

// Visual-only: where does a given DP row sit on the canvas once we've added
// gutters between subpartitions? Returns the y-pixel of the top of `dp`.
function yOfDp(dp: number): number {
  const sub = Math.floor(dp / DP_PER_SUB);
  const within = dp % DP_PER_SUB;
  return sub * (DP_PER_SUB * (CELL + 1) + GUTTER) + within * (CELL + 1);
}

function subBlockHeight(): number {
  return DP_PER_SUB * (CELL + 1);
}

function totalHeight(): number {
  return TMEM_SUBPARTITIONS * subBlockHeight() + (TMEM_SUBPARTITIONS - 1) * GUTTER;
}

// Per-subpartition port activity. We schematically saturate a subpartition
// proportional to how much of its DP range the accumulator footprint covers
// times the current `fill` level. This is honest: we do not model port
// latency, just which subpartitions are currently exchanging data.
function subActivity(sub: number, accDp: number, fill: number): number {
  const start = sub * DP_PER_SUB;
  const end = start + DP_PER_SUB;
  const overlap = Math.max(0, Math.min(end, accDp) - start);
  return (overlap / DP_PER_SUB) * fill;
}

export function TmemPanel() {
  const i = inst.value;
  const cCons = currentConsumerPhase.value;
  const cEpi = currentEpiloguePhase.value;
  const cProg = phaseProgress.value; // consumer-priority (matches mma pulse)
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [shapeId, setShapeId] = useState<string>('16dp256b x4');
  const [peer, setPeer] = useState<0 | 1>(0);

  const usesTmem = i.accIn === 'tmem' || i.aSource.includes('tmem');
  const ctaGroup = (i.ctaGroup ?? 1) as 1 | 2;
  // Phase 6: TS variant lives-here-in-TMEM region drives an extra highlight
  // band during tcgen05.cp producer phases. cg2 peer visibility is bound to
  // world.cluster?.peerActive so it dims during idle ticks.
  const w = world.value;
  const sumV = summary.value;
  const isTs = sumV.variant === 'ts';
  const isCg2 = sumV.extras.ctaGroup === 2;
  const tsCopyActive = w.producerTransfer?.kind === 'tcgen05-cp';
  const peerActive = isCg2 ? !!w.cluster?.peerActive : false;
  // Derive kStep from world.consumerAtom for the mma.step fill calculation.
  const consAtom = w.consumerAtom;
  const kStep = consAtom
    ? { k: consAtom.kStep, total: sumV.consumerItersTotal }
    : null;
  const ptxLayout = usesTmem
    ? classifyLayout(i.M, ctaGroup, { sparse: i.sparse, warpSpecialized: i.warpSpecialized })
    : null;
  const layoutInfo = ptxLayout ? PTX_LAYOUTS[ptxLayout] : null;
  const baseFootprint = accFootprint(i.M, i.N, ctaGroup);
  const accDp = layoutInfo ? Math.min(TMEM_DP, layoutInfo.dpRows) : baseFootprint.dp;
  const accCols = baseFootprint.cols;
  const range = ptxLayout ? peerMRange(ptxLayout, peer, i.M) : null;

  // Accumulator fill: ramps linearly as each mma.step lands a k-atom of
  // contribution into TMEM. Goes to full during tcgen05.ld (reading it out)
  // and epilogue.tma.store (we already ran every k-step). Driven from
  // world so the animation matches the consumer's actual progress (which
  // under warpspec is offset from the producer by up to kStages-1 slabs).
  const fill = (() => {
    if (cCons?.kind === 'tcgen05.mma.step' && kStep) {
      return Math.min(1, (kStep.k + cProg) / kStep.total);
    }
    const epiKind = cEpi?.kind;
    if (epiKind === 'tcgen05.ld' || epiKind === 'epilogue.stg_smem' || epiKind === 'epilogue.tma.store') {
      return 1;
    }
    // Outside mma / epilogue (e.g. pure warmup producer-only ticks): hold
    // whatever we last computed. For simplicity return the running fraction
    // of completed mma steps from cTile.accumulated; normalises smoothly.
    const acc = w.cTile?.accumulated?.[0]?.[0] ?? 0;
    return Math.min(1, acc / sumV.consumerItersTotal);
  })();

  const SHOWN_COLS = 128;
  const W = SHOWN_COLS * (CELL + 1);
  const H = totalHeight();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = `${W}px`;
    canvas.style.height = `${H}px`;
    const ctx = canvas.getContext('2d')!;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    // 1. Four subpartition blocks with gutters.
    for (let sub = 0; sub < TMEM_SUBPARTITIONS; sub++) {
      ctx.fillStyle = subColor(sub);
      ctx.fillRect(0, yOfDp(sub * DP_PER_SUB), W, subBlockHeight());
      // Subpartition label band at the very top of each block.
      ctx.fillStyle = 'rgba(0,0,0,.35)';
      ctx.fillRect(0, yOfDp(sub * DP_PER_SUB), 18, 10);
      ctx.fillStyle = '#cfd4e0';
      ctx.font = '8px "JetBrains Mono", Menlo, monospace';
      ctx.fillText(`s${sub}`, 2, yOfDp(sub * DP_PER_SUB) + 8);
    }

    // 2. Accumulator footprint — bright overlay; spans contiguous DP rows
    //    0..accDp which may cross subpartition boundaries (with gutters).
    if (usesTmem) {
      const alpha = 0.15 + 0.55 * fill;
      // Draw per-subpartition slice so gutters aren't shaded.
      for (let sub = 0; sub < TMEM_SUBPARTITIONS; sub++) {
        const start = sub * DP_PER_SUB;
        const end = start + DP_PER_SUB;
        const overlapStart = Math.max(start, 0);
        const overlapEnd = Math.min(end, accDp);
        if (overlapEnd <= overlapStart) continue;
        const yTop = yOfDp(overlapStart);
        const hPx = (overlapEnd - overlapStart) * (CELL + 1);
        const cw = Math.min(accCols, SHOWN_COLS) * (CELL + 1);
        ctx.fillStyle = `rgba(255, 200, 120, ${alpha})`;
        ctx.fillRect(0, yTop, cw, hPx);
        ctx.strokeStyle = '#ffc878';
        ctx.lineWidth = 1;
        ctx.strokeRect(0 + 0.5, yTop + 0.5, cw - 1, hPx - 1);
      }
      // Leading edge of the fill across the whole footprint (swept along cols).
      if (fill > 0 && fill < 1) {
        const cwFull = Math.min(accCols, SHOWN_COLS) * (CELL + 1);
        const edgeX = Math.floor(cwFull * fill);
        const yTop = yOfDp(0);
        const hPx = yOfDp(Math.max(0, accDp - 1)) + (CELL + 1) - yTop;
        ctx.fillStyle = `rgba(255, 220, 140, ${0.4 + 0.4 * fill})`;
        ctx.fillRect(edgeX - 2, yTop, 3, hPx);
      }
    }

    // 3. Peer CTA footprint (cta_group::2) — draw overlaid on the SAME DP
    //    range as our own acc (the pair shares TMEM rows) but distinguished
    //    by a dashed outline. Phase 6: opacity follows world.cluster.peerActive
    //    so the dashed outline fades during idle ticks.
    if (usesTmem && ctaGroup === 2 && range) {
      const otherPeer: 0 | 1 = peer === 0 ? 1 : 0;
      const otherRange = peerMRange(ptxLayout!, otherPeer, i.M);
      if (otherRange) {
        const cw = Math.min(accCols, SHOWN_COLS) * (CELL + 1);
        const peerAlpha = peerActive ? 0.85 : 0.35;
        ctx.strokeStyle = `rgba(180, 142, 173, ${peerAlpha})`;
        ctx.setLineDash([4, 3]);
        ctx.lineWidth = 1.5;
        for (let sub = 0; sub < TMEM_SUBPARTITIONS; sub++) {
          const start = sub * DP_PER_SUB;
          const end = start + DP_PER_SUB;
          const overlapStart = Math.max(start, 0);
          const overlapEnd = Math.min(end, accDp);
          if (overlapEnd <= overlapStart) continue;
          const yTop = yOfDp(overlapStart);
          const hPx = (overlapEnd - overlapStart) * (CELL + 1);
          ctx.strokeRect(2.5, yTop + 2.5, cw - 5, hPx - 5);
        }
        ctx.setLineDash([]);
        // Small label in the top-right corner so the reader knows what the
        // dashed outline represents.
        ctx.fillStyle = `rgba(180, 142, 173, ${peerAlpha})`;
        ctx.font = '9px "JetBrains Mono", Menlo, monospace';
        ctx.fillText(
          `peer ${otherPeer === 0 ? 'CTA₀' : 'CTA₁'} M[${otherRange.lo}..${otherRange.hi})` +
            (peerActive ? '' : ' · idle'),
          cw + 6,
          yOfDp(0) + 10,
        );
      }
    }

    // Phase 6 — TS variant: "A region" in TMEM lights up when tcgen05.cp is
    // copying SMEM-A → TMEM. Draw a small purple band ABOVE the accumulator
    // region (in DP 0..accDp, column offset >> acc columns to not overlap).
    if (usesTmem && isTs) {
      const aRegionCols = Math.min(64, SHOWN_COLS - Math.min(accCols, SHOWN_COLS));
      if (aRegionCols > 0) {
        const xStart = Math.min(accCols, SHOWN_COLS) * (CELL + 1) + 4;
        const cw = aRegionCols * (CELL + 1);
        // Band spans just the DP range the acc uses (same rows — A shares rows
        // with C atom, different column range is schematic since real TMEM
        // allocation differs).
        const yTop = yOfDp(0);
        const hPx = yOfDp(Math.max(0, accDp - 1)) + (CELL + 1) - yTop;
        const alpha = tsCopyActive ? 0.55 : 0.15;
        // Use --mbar purple (#b48ead).
        ctx.fillStyle = `rgba(180, 142, 173, ${alpha})`;
        ctx.fillRect(xStart, yTop, cw, hPx);
        ctx.strokeStyle = `rgba(180, 142, 173, ${tsCopyActive ? 1.0 : 0.5})`;
        ctx.lineWidth = 1.5;
        ctx.strokeRect(xStart + 0.5, yTop + 0.5, cw - 1, hPx - 1);
        ctx.fillStyle = '#b48ead';
        ctx.font = '9px "JetBrains Mono", Menlo, monospace';
        ctx.fillText(
          tsCopyActive ? 'A region · tcgen05.cp active' : 'A region (TS)',
          xStart + 4,
          yTop + 11,
        );
      }
    }

    // 4. tcgen05.ld/.st shape overlay — draw ONE issue's exact footprint,
    //    then `repeat` replicas tiled horizontally.
    const shape = findShape(shapeId);
    if (shape && usesTmem) {
      const colsPerIssue = shape.bits / 32;
      for (let rep = 0; rep < shape.repeat; rep++) {
        const xStart = rep * colsPerIssue * (CELL + 1);
        if (xStart >= W) break;
        const wPx = colsPerIssue * (CELL + 1);
        const yTop = yOfDp(0);
        const hPx = shape.dp * (CELL + 1);
        ctx.strokeStyle = '#7e9cd8';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(xStart + 0.5, yTop + 0.5, wPx - 1, hPx - 1);
        ctx.fillStyle = `rgba(126, 156, 216, ${0.12 + 0.06 * (rep % 2)})`;
        ctx.fillRect(xStart, yTop, Math.min(wPx, W - xStart), hPx);
      }
    }
  }, [i.id, usesTmem, accDp, accCols, shapeId, fill, ctaGroup, peer, W, H, range?.lo, range?.hi, ptxLayout, isTs, tsCopyActive, peerActive]);

  return (
    <div class="panel tmem-panel">
      <h3>
        TMEM <small>— 4 subp × {DP_PER_SUB} DP × {TMEM_COLS} cols (sm_100 only)</small>
      </h3>
      {usesTmem ? (
        <div class="tmem__stage">
          <canvas ref={canvasRef} style={{ height: H }} />
          {/* Per-subpartition port activity meter */}
          <div class="tmem__ports" style={{ height: H }}>
            {Array.from({ length: TMEM_SUBPARTITIONS }, (_, sub) => {
              const act = subActivity(sub, accDp, fill);
              return (
                <div
                  key={sub}
                  class="tmem__port"
                  style={{
                    top: `${yOfDp(sub * DP_PER_SUB)}px`,
                    height: `${subBlockHeight()}px`,
                  }}
                  title={`subp ${sub} · activity ${(act * 100).toFixed(0)}%`}
                >
                  <div class="tmem__port-bar" style={{ width: `${act * 100}%` }} />
                  <span class="tmem__port-lbl">s{sub}</span>
                </div>
              );
            })}
          </div>
        </div>
      ) : (
        <div class="tmem-placeholder tmem-placeholder--dim">
          <code>{i.family}</code> does not use TMEM — the accumulator lives in{' '}
          <code>.reg</code> (see the SIMT registers panel below).
          <br />
          TMEM is sm_100 only; switch to a <code>tcgen05.*</code> instruction to see
          the 128 DP × 512 cols accumulator memory.
        </div>
      )}
      {usesTmem ? (
        <>
          <p class="panel__note">
            acc footprint: <code>{accDp} DP × {accCols} cols</code>
            {ctaGroup === 2 && ' (cta_group::2 halves M across 2 CTAs)'}
          </p>
          {layoutInfo ? (
            <p class="panel__note">
              PTX <code>{layoutInfo.name}</code>:{' '}
              <span class="tmem__conditions">{layoutInfo.conditions}</span>
              <span class="tmem__desc">{layoutInfo.description}</span>
            </p>
          ) : (
            <p class="panel__note panel__note--dim">
              (M={i.M}, cta={ctaGroup} not in PTX A–G taxonomy)
            </p>
          )}
          {range && (
            <p class="panel__note">
              viewing{' '}
              <code>{range.peerLabel}</code> · holds <code>M[{range.lo}..{range.hi})</code>
              {ctaGroup === 2 && (
                <>
                  {' '}
                  <button
                    class="tmem__peer-btn"
                    onClick={() => setPeer((p) => (p === 0 ? 1 : 0))}
                  >
                    switch peer
                  </button>
                </>
              )}
            </p>
          )}
          <label class="tmem__pick">
            tcgen05 shape
            <select value={shapeId} onChange={(e) => setShapeId((e.target as HTMLSelectElement).value)}>
              {TCGEN05_SHAPES.map((s) => (
                <option value={s.id}>{s.id}</option>
              ))}
            </select>
          </label>
          <div class="tmem__legend">
            <span class="tmem__swatch" style={{ background: subColor(0) }} /> subp 0
            <span class="tmem__swatch" style={{ background: subColor(1) }} /> subp 1
            <span class="tmem__swatch" style={{ background: subColor(2) }} /> subp 2
            <span class="tmem__swatch" style={{ background: subColor(3) }} /> subp 3
            <span class="tmem__swatch tmem__swatch--acc" /> acc
            <span class="tmem__swatch tmem__swatch--issue" /> ld/st issue
            {ctaGroup === 2 && (
              <>
                <span class="tmem__swatch tmem__swatch--peer" /> peer CTA
              </>
            )}
          </div>
        </>
      ) : (
        <p class="panel__note panel__note--dim">
          {i.family} does not use TMEM (accumulator lives in registers / <code>.reg</code>).
        </p>
      )}
      <TruthFooter
        models="TMEM geometry (128 DP × 512 cols for sm_100); accumulator footprint from accFootprint(M, N, ctaGroup); per-subpartition occupancy indicator; TS A-region highlighted during tcgen05.cp; cg2 peer footprint when world.cluster.peerActive."
        schematic="exact port latency; allocator state; cross-SM TMEM coherence; real A-region location is allocator-dependent."
        cite="simulation.ts · world.producerTransfer / world.cluster; tmem.ts · accFootprint"
      />
    </div>
  );
}
