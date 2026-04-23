// Tile hierarchy panel — the single visual that teaches the CUTLASS tile
// story: Problem → ClusterShape → TileShape(BLK_M, BLK_N, BLK_K) → PIPE × K
// slots → TiledMMA atoms. One SVG with three stacked rows:
//
//   Row 1 — GMEM problem (schematic). This CTA's tile is highlighted; for
//           clusterShape[0] > 1 the peer CTAs share the same cluster row.
//   Row 2 — CTA K-walk. numIters = ceil(BLK_K_total / atom_K) slots laid
//           out horizontally; coloured by role (consume / in-flight / drained
//           / pending) derived from `ringState(iter, kStages)`. A consumer
//           cursor tracks the active K iteration.
//   Row 3 — CTA tile with MMA-atom grid. `blkMMult × blkNMult` atom cells;
//           the currently-executing atom is highlighted.
//
// Scrubbing the Timeline updates all three rows. Clicking a Row 2 slot jumps
// the tick to the iteration that consumes it.
//
// Model boundaries (TruthFooter): we model integer atom multiples only
// (AtomLayoutMNK with M/N multipliers in {1,2,4}) and a simplified 4×4 GMEM
// schematic. Real CUTLASS kernels can use `TiledMMA<AtomLayoutMNK, ValLayoutMNK>`
// for richer partitioning.

import { useMemo } from 'preact/hooks';
import {
  atomsPerCta,
  blkMMult,
  blkNMult,
  clusterShape,
  consumerPhases,
  ctaGrid,
  currentKStep,
  currentProducerKStep,
  inst,
  kStages,
  phases,
  pipelineMode,
  problemMMult,
  problemNMult,
  problemKMult,
  summary,
  tick,
  tileK,
  world,
} from '../state';
import { ringState } from '../pipeline_state';
import { TruthFooter } from './TruthFooter';

const W = 880;
const ROW_H = 110;
const GAP = 10;
const LEFT_GUTTER = 120;

// Row 1 — real GMEM problem geometry from `ctaGrid` (user-configurable via
// ConfigBar's PROBLEM group). The demo CTA sits at (0, 0); the cluster peer
// sits at (1, 0) on the M axis when clusterM > 1.

function roleColor(role: 'consume' | 'fill' | 'hold' | 'drained' | 'pending'): string {
  switch (role) {
    case 'consume': return '#ffd878';
    case 'fill': return '#7ec699';
    case 'hold': return '#5a6378';
    case 'drained': return '#2a2f3a';
    case 'pending': return '#1a1f2a';
  }
}

export function TileHierarchyPanel() {
  const i = inst.value;
  const bm = blkMMult.value;
  const bn = blkNMult.value;
  const { m: atomsM, n: atomsN } = atomsPerCta.value;
  const [clusterM, , ] = clusterShape.value;
  const stages = kStages.value;
  const tK = tileK.value;
  // Phase 6 — variant overlays driven by summary + world.cluster.
  const sumV = summary.value;
  const isCg2 = sumV.extras.ctaGroup === 2;
  const isSparse = sumV.extras.sparse;
  const isBlockScaled = sumV.extras.blockScaled;
  const clusterPeerActive = isCg2 ? !!world.value.cluster?.peerActive : false;
  // Row 2 K-walk slot count is now the total consumer iters
  // (slabCount × atomsPerStage_K) per Phase 2 — matches what the simulator
  // emits, and reflects the true nested outer-slab × inner-k-atom structure.
  const numIters = summary.value.consumerItersTotal;
  const kStep = currentKStep.value;
  const pStep = currentProducerKStep.value;
  const phs = phases.value;
  const grid = ctaGrid.value;
  const pkm = problemKMult.value;
  const mode = pipelineMode.value;
  void consumerPhases;
  void problemMMult;
  void problemNMult;

  // Which K iter is currently being consumed (-1 if we're not in an MMA step).
  const consumerIter = kStep?.k ?? -1;
  // Producer head — the K slice the producer warpgroup is currently filling.
  // In warpspec steady state this leads the consumer by (kStages − 1); in
  // coupled regime it collapses onto the consumer.
  const producerIter = pStep?.k ?? (mode === 'coupled' ? consumerIter : -1);

  // Map K slot index → role + slice id (for tooltip).
  const slotRoles = useMemo(() => {
    const out: { role: 'consume' | 'fill' | 'hold' | 'drained' | 'pending'; slice: number }[] = [];
    if (consumerIter < 0) {
      // Outside an MMA step — mark everything "pending" to keep the row quiet.
      for (let k = 0; k < numIters; k++) out.push({ role: 'pending', slice: k });
      return out;
    }
    const ring = ringState(consumerIter, stages);
    const ringSlices = new Set(ring.map((r) => r.slice));
    const consumerSlice = ring.find((r) => r.role === 'consume')?.slice ?? consumerIter;
    const producerSlice = ring.find((r) => r.role === 'fill')?.slice ?? consumerSlice + stages - 1;
    for (let k = 0; k < numIters; k++) {
      if (k < consumerSlice) out.push({ role: 'drained', slice: k });
      else if (k === consumerSlice) out.push({ role: 'consume', slice: k });
      else if (ringSlices.has(k) && k === producerSlice) out.push({ role: 'fill', slice: k });
      else if (ringSlices.has(k)) out.push({ role: 'hold', slice: k });
      else out.push({ role: 'pending', slice: k });
    }
    return out;
  }, [consumerIter, stages, numIters]);

  // Which atom (atomM × atomN index) is currently executing. Driven by the
  // simulator's `world.consumerAtom`: during an active consumer phase this
  // is the MN atom currently firing (m-outer, n-inner order per plan §A).
  // Outside a consumer phase `consumerAtom === null` → no atom highlighted.
  const cAtom = world.value.consumerAtom;
  const activeAtom = useMemo<{ m: number; n: number } | null>(() => {
    return cAtom ? { m: cAtom.atomM, n: cAtom.atomN } : null;
  }, [cAtom]);

  // --- Row 1: GMEM problem geometry ---
  const row1 = 0;
  // Show a (rowsM × colsN) slab of CTA tiles, at K=0 slice. K is handled by
  // Row 2's K-walk.
  const gridCols = Math.max(1, grid.colsN);
  const gridRows = Math.max(1, grid.rowsM);
  const cellW = (W - LEFT_GUTTER - 20) / gridCols;
  const cellH = (ROW_H - 30) / gridRows;
  // "This" CTA sits at (0, 0); cluster peer at (1, 0) on the M axis.
  const myCx = 0, myCy = 0;

  // --- Row 2: CTA K-walk ---
  const row2 = ROW_H + GAP;
  const walkW = W - LEFT_GUTTER - 20;
  const slotW = walkW / Math.max(1, numIters);

  // --- Row 3: CTA tile subdivided into atoms ---
  const row3 = 2 * (ROW_H + GAP);
  const tileRowH = ROW_H - 30;
  const tileRowW = Math.min(walkW, 420);
  void tileRowW;

  const onJumpToSlot = (k: number) => {
    // Find the first phase of kind wgmma.step / tcgen05.mma.step whose iter index equals k.
    const steps = phs.filter((p) => p.kind === 'wgmma.step' || p.kind === 'tcgen05.mma.step');
    const target = steps[k] ?? steps[steps.length - 1];
    if (target) tick.value = target.startTick + 0.001;
  };

  return (
    <div class="panel tile-hier">
      <h3>
        Tile hierarchy
        <small>
          — <code>TileShape = ({i.M * bm}, {i.N * bn}, {tK})</code> ·{' '}
          <code>ClusterShape = ({clusterM},1,1)</code> ·{' '}
          <code>Atom = ({i.M}, {i.N}, {i.K})</code> ·{' '}
          <code>PIPE = {stages}</code> ·{' '}
          <code>numIters = {numIters}</code>
        </small>
      </h3>

      <svg class="tile-hier__svg" width="100%" viewBox={`0 0 ${W} ${row3 + tileRowH + 30}`}>
        {/* Row 1 label & GMEM grid */}
        <text x={8} y={row1 + 14} class="tile-hier__rowlabel">GMEM problem</text>
        <text x={8} y={row1 + 28} class="tile-hier__rowlabel tile-hier__rowlabel--dim">
          {gridRows} × {gridCols} CTA tiles
          {pkm > 1 && ` × ${pkm} K slabs`}
        </text>
        <g transform={`translate(${LEFT_GUTTER}, ${row1 + 4})`}>
          {Array.from({ length: gridRows }, (_, gy) =>
            Array.from({ length: gridCols }, (_, gx) => {
              const isMe = gx === myCx && gy === myCy;
              // Cluster peer: the CTA that shares the cluster along M with us.
              const isPeer = clusterM > 1 && gx === myCx && gy === myCy + 1;
              return (
                <rect
                  key={`g-${gx}-${gy}`}
                  x={gx * cellW}
                  y={gy * cellH}
                  width={cellW - 2}
                  height={cellH - 2}
                  class={`tile-hier__gcell ${isMe ? 'is-me' : ''} ${isPeer ? 'is-peer' : ''}`}
                />
              );
            }),
          )}
          {/* Cluster outline */}
          {clusterM > 1 && (
            <rect
              x={myCx * cellW - 2}
              y={myCy * cellH - 2}
              width={cellW + 2}
              height={2 * cellH + 2}
              class="tile-hier__cluster"
            />
          )}
          {/* Annotate */}
          <text x={myCx * cellW + cellW / 2 - 1} y={myCy * cellH + cellH / 2 + 4} text-anchor="middle" class="tile-hier__celllbl">
            this CTA
          </text>
          {clusterM > 1 && gridRows > 1 && (
            <text x={myCx * cellW + cellW / 2 - 1} y={(myCy + 1) * cellH + cellH / 2 + 4} text-anchor="middle" class="tile-hier__celllbl tile-hier__celllbl--peer">
              peer
            </text>
          )}
        </g>

        {/* Row 2 label & K-walk */}
        <text x={8} y={row2 + 14} class="tile-hier__rowlabel">CTA K-walk</text>
        <text x={8} y={row2 + 28} class="tile-hier__rowlabel tile-hier__rowlabel--dim">
          {numIters} × atom_K={i.K}
        </text>
        <g transform={`translate(${LEFT_GUTTER}, ${row2 + 10})`}>
          {slotRoles.map((s, k) => (
            <g key={`slot-${k}`}>
              <rect
                x={k * slotW}
                y={0}
                width={slotW - 2}
                height={tileRowH}
                fill={roleColor(s.role)}
                class={`tile-hier__slot tile-hier__slot--${s.role}`}
                onClick={() => onJumpToSlot(k)}
              >
                <title>k={s.slice} · {s.role}</title>
              </rect>
              <text
                x={k * slotW + slotW / 2 - 1}
                y={tileRowH / 2 + 4}
                text-anchor="middle"
                class="tile-hier__slotlbl"
                style={{ fill: s.role === 'consume' || s.role === 'fill' ? '#0b0e14' : '#8b96a8' }}
              >
                k{k}
              </text>
            </g>
          ))}
          {/* Consumer cursor */}
          {consumerIter >= 0 && (
            <line
              x1={consumerIter * slotW + slotW / 2 - 1}
              x2={consumerIter * slotW + slotW / 2 - 1}
              y1={-4}
              y2={tileRowH + 4}
              class="tile-hier__cursor"
            />
          )}
          {/* Producer cursor — dotted, leads the consumer in steady state.
              Hidden in coupled mode because producer and consumer are on the
              same step. */}
          {mode === 'warpspec' && producerIter >= 0 && producerIter !== consumerIter && (
            <>
              <line
                x1={producerIter * slotW + slotW / 2 - 1}
                x2={producerIter * slotW + slotW / 2 - 1}
                y1={-4}
                y2={tileRowH + 4}
                class="tile-hier__cursor tile-hier__cursor--producer"
                stroke-dasharray="4 2"
              />
              <text
                x={producerIter * slotW + slotW / 2 - 1}
                y={-8}
                text-anchor="middle"
                class="tile-hier__cursorlbl"
              >
                prod
              </text>
            </>
          )}
          {consumerIter >= 0 && (
            <text
              x={consumerIter * slotW + slotW / 2 - 1}
              y={tileRowH + 18}
              text-anchor="middle"
              class="tile-hier__cursorlbl"
            >
              cons
            </text>
          )}
        </g>

        {/* Row 3 — CTA tile hierarchy:  A (BLK_M × BLK_K)  +  B (BLK_K × BLK_N)  =  C (BLK_M × BLK_N).
            Each operand is drawn at the same vertical scale for BLK_M/BLK_N
            and a fixed K width so K can be compared across operands.
            Atoms are shown as faint subdivisions inside each tile. */}
        <text x={8} y={row3 + 14} class="tile-hier__rowlabel">CTA tile hierarchy</text>
        <text x={8} y={row3 + 28} class="tile-hier__rowlabel tile-hier__rowlabel--dim">
          A + B ⇒ C · TiledMMA<tspan baseline-shift="sub">{atomsM}×{atomsN}</tspan>
        </text>
        {(() => {
          // Three sub-tiles laid out side by side: A, B, C.
          // Available horizontal room: walkW (numbers below LEFT_GUTTER).
          const avail = walkW;
          // Reserve a small budget for the two "⇒" connectors.
          const conn = 26;
          const totalOps = avail - 2 * conn;
          // Give C slightly more room since it's the final output.
          const scale = totalOps / (Math.max(tK, 1) + Math.max(i.N * bn, 1) + Math.max(i.N * bn, 1));
          const aW = Math.max(14, Math.floor(tK * scale));           // BLK_K cells wide
          const bW = Math.max(20, Math.floor(i.N * bn * scale));     // BLK_N cells wide
          const cW = Math.max(20, Math.floor(i.N * bn * scale));     // BLK_N cells wide
          // Height: A is BLK_M tall, B is BLK_K tall, C is BLK_M tall.
          const baseH = tileRowH;
          const aH = baseH;
          const bH = Math.max(14, Math.floor(baseH * Math.min(1, tK / (i.M * bm))));
          const cH = baseH;
          let cursor = 0;

          const aX = cursor; cursor += aW + conn;
          const bX = cursor; cursor += bW + conn;
          const cX = cursor;

          // Active atom on C maps back to which atoms of A/B it reads from:
          // C[am,an] += A[am,K] · B[K,an] (K is the mma.step iter).
          const activeA = activeAtom ? { m: activeAtom.m, k: 0 } : null;
          const activeB = activeAtom ? { k: 0, n: activeAtom.n } : null;

          return (
            <g transform={`translate(${LEFT_GUTTER}, ${row3 + 4})`}>
              {/* A tile */}
              <g transform={`translate(${aX}, 0)`}>
                <rect x={0} y={0} width={aW} height={aH} class="tile-hier__ctatile tile-hier__ctatile--A" />
                {/* atom subdivisions along M (BLK_M direction) */}
                {Array.from({ length: atomsM }, (_, am) => {
                  const active = activeA?.m === am && consumerIter >= 0;
                  return (
                    <rect
                      key={`atomA-${am}`}
                      x={1}
                      y={am * (aH / atomsM) + 1}
                      width={aW - 2}
                      height={aH / atomsM - 2}
                      class={`tile-hier__atom tile-hier__atom--A ${active ? 'is-active' : ''}`}
                    />
                  );
                })}
                <text x={aW / 2} y={-6} text-anchor="middle" class="tile-hier__axislbl">A</text>
                <text x={aW / 2} y={aH + 12} text-anchor="middle" class="tile-hier__axislbl tile-hier__axislbl--dim">
                  {i.M * bm} × {tK}
                </text>
              </g>
              {/* "+" connector between A and B */}
              <text x={aX + aW + conn / 2} y={aH / 2 + 6} text-anchor="middle" class="tile-hier__op">×</text>

              {/* B tile */}
              <g transform={`translate(${bX}, 0)`}>
                <rect x={0} y={0} width={bW} height={bH} class="tile-hier__ctatile tile-hier__ctatile--B" />
                {Array.from({ length: atomsN }, (_, an) => {
                  const active = activeB?.n === an && consumerIter >= 0;
                  return (
                    <rect
                      key={`atomB-${an}`}
                      x={an * (bW / atomsN) + 1}
                      y={1}
                      width={bW / atomsN - 2}
                      height={bH - 2}
                      class={`tile-hier__atom tile-hier__atom--B ${active ? 'is-active' : ''}`}
                    />
                  );
                })}
                <text x={bW / 2} y={-6} text-anchor="middle" class="tile-hier__axislbl">B</text>
                <text x={bW / 2} y={bH + 12} text-anchor="middle" class="tile-hier__axislbl tile-hier__axislbl--dim">
                  {tK} × {i.N * bn}
                </text>
              </g>
              {/* "⇒" connector between B and C */}
              <text x={bX + bW + conn / 2} y={cH / 2 + 6} text-anchor="middle" class="tile-hier__op">⇒</text>

              {/* C tile with atom grid — Phase 5: per-atom opacity reflects
                  cTile.accumulated[am][an] / maxAccumulatedPerAtom so the
                  cells literally fill up as k-steps land on them. Active
                  atom (from consumerAtom) gets a brighter overlay stroke.
                  Fully-accumulated cells get a small check glyph. */}
              <g transform={`translate(${cX}, 0)`}>
                <rect x={0} y={0} width={cW} height={cH} class="tile-hier__ctatile tile-hier__ctatile--C" />
                {Array.from({ length: atomsM }, (_, am) =>
                  Array.from({ length: atomsN }, (_, an) => {
                    const isActive = activeAtom && activeAtom.m === am && activeAtom.n === an;
                    const maxAcc = summary.value.maxAccumulatedPerAtom;
                    const acc = world.value.cTile?.accumulated[am]?.[an] ?? 0;
                    const fillOpacity = maxAcc > 0 ? Math.min(1, acc / maxAcc) : 0;
                    const isFull = maxAcc > 0 && acc >= maxAcc;
                    const cx0 = an * (cW / atomsN) + 1;
                    const cy0 = am * (cH / atomsM) + 1;
                    const cwCell = cW / atomsN - 2;
                    const chCell = cH / atomsM - 2;
                    return (
                      <g key={`atomC-${am}-${an}`}>
                        {/* Green accumulation fill layered over the atom cell. */}
                        {fillOpacity > 0 && (
                          <rect
                            x={cx0}
                            y={cy0}
                            width={cwCell}
                            height={chCell}
                            fill="#7ec699"
                            opacity={fillOpacity}
                          >
                            <title>
                              C atom ({am},{an}) accumulated {acc}/{maxAcc}
                            </title>
                          </rect>
                        )}
                        <rect
                          x={cx0}
                          y={cy0}
                          width={cwCell}
                          height={chCell}
                          class={`tile-hier__atom ${isActive ? 'is-active' : ''}`}
                          fill="transparent"
                        >
                          <title>
                            C atom ({am},{an}) · {i.M}×{i.N} · accumulated {acc}/{maxAcc}
                          </title>
                        </rect>
                        {/* Full check glyph for completed atoms. */}
                        {isFull && (
                          <text
                            x={cx0 + cwCell / 2}
                            y={cy0 + chCell / 2 + 3}
                            text-anchor="middle"
                            class="tile-hier__celllbl"
                            style={{ fill: '#0b0e14', pointerEvents: 'none' }}
                          >
                            ✓
                          </text>
                        )}
                      </g>
                    );
                  }),
                )}
                <text x={cW / 2} y={-6} text-anchor="middle" class="tile-hier__axislbl">C (acc)</text>
                <text x={cW / 2} y={cH + 12} text-anchor="middle" class="tile-hier__axislbl tile-hier__axislbl--dim">
                  {i.M * bm} × {i.N * bn}
                </text>
                {/* Phase 6 — cg2: split C tile horizontally. Our half is
                    the solid-bordered body above; the peer half is a dashed
                    overlay mirroring the same accumulation grid. The peer
                    executes the same MMAs in lockstep (A-split), so mirror
                    the opacity from our cTile.accumulated. */}
                {isCg2 && (
                  <g class="tile-hier__cg2-peer">
                    <rect
                      x={0}
                      y={cH + 18}
                      width={cW}
                      height={cH}
                      fill="transparent"
                      stroke={clusterPeerActive ? '#b48ead' : '#5a6378'}
                      stroke-dasharray="4 3"
                      stroke-width={1.5}
                      opacity={clusterPeerActive ? 1 : 0.45}
                    />
                    {Array.from({ length: atomsM }, (_, am) =>
                      Array.from({ length: atomsN }, (_, an) => {
                        const maxAcc = summary.value.maxAccumulatedPerAtom;
                        const acc = world.value.cTile?.accumulated[am]?.[an] ?? 0;
                        const fillOpacity = maxAcc > 0 ? Math.min(1, acc / maxAcc) : 0;
                        if (fillOpacity <= 0) return null;
                        const cx0 = an * (cW / atomsN) + 1;
                        const cy0 = (cH + 18) + am * (cH / atomsM) + 1;
                        const cwCell = cW / atomsN - 2;
                        const chCell = cH / atomsM - 2;
                        return (
                          <rect
                            key={`peerC-${am}-${an}`}
                            x={cx0}
                            y={cy0}
                            width={cwCell}
                            height={chCell}
                            fill="#b48ead"
                            opacity={fillOpacity * (clusterPeerActive ? 0.55 : 0.2)}
                          />
                        );
                      }),
                    )}
                    <text
                      x={cW / 2}
                      y={cH + 18 + cH / 2 + 3}
                      text-anchor="middle"
                      class="tile-hier__axislbl tile-hier__axislbl--dim"
                      style={{ fill: '#b48ead', pointerEvents: 'none' }}
                    >
                      peer CTA half
                    </text>
                  </g>
                )}
              </g>

              {/* Phase 6 — sparse badge on A tile (2:4 metadata). */}
              {isSparse && (
                <g class="tile-hier__sparse-badge" transform={`translate(${aX}, 0)`}>
                  <rect
                    x={2}
                    y={2}
                    width={26}
                    height={12}
                    rx={2}
                    fill="#b48ead"
                    opacity={0.85}
                  />
                  <text
                    x={15}
                    y={11}
                    text-anchor="middle"
                    class="tile-hier__axislbl"
                    style={{ fill: '#0b0e14', fontSize: '8px', pointerEvents: 'none' }}
                  >
                    2:4 sp
                  </text>
                </g>
              )}

              {/* Phase 6 — block_scaled: SFA chip on A, SFB chip on B. */}
              {isBlockScaled && (
                <>
                  <g transform={`translate(${aX}, 0)`}>
                    <rect x={2} y={2} width={26} height={12} rx={2} fill="#e0cf7a" opacity={0.85} />
                    <text x={15} y={11} text-anchor="middle" class="tile-hier__axislbl"
                      style={{ fill: '#0b0e14', fontSize: '8px', pointerEvents: 'none' }}>
                      SFA
                    </text>
                  </g>
                  <g transform={`translate(${bX}, 0)`}>
                    <rect x={2} y={2} width={26} height={12} rx={2} fill="#e0cf7a" opacity={0.85} />
                    <text x={15} y={11} text-anchor="middle" class="tile-hier__axislbl"
                      style={{ fill: '#0b0e14', fontSize: '8px', pointerEvents: 'none' }}>
                      SFB
                    </text>
                  </g>
                </>
              )}
            </g>
          );
        })()}
        {/* axis guidance on the far left */}
        <text x={LEFT_GUTTER - 8} y={row3 + 4 + tileRowH / 2 + 4} text-anchor="end" class="tile-hier__axislbl">
          BLK_M axis →
        </text>
      </svg>

      <TruthFooter
        models="CUTLASS TileShape (BLK_M, BLK_N, BLK_K), ClusterShape, TiledMMA atom grid, problem-level CTA grid from ctaGrid; nested PIPE × slab × k-atom K-walk with dual producer/consumer cursors under warpspec; Row 3 atom fill opacity follows world.cTile.accumulated."
        schematic="atom iteration order within an mma.step is m-outer/n-inner per CUTLASS default but real kernels can pick different AtomLayoutMNK; CTA coordinate fixed to (0,0); cg2 peer C-tile is a mirrored schematic of ours (simulator tracks one CTA)."
        cite="cute/atom/atom.hpp · TiledMMA; cute/algorithm/tile.hpp · local_tile"
      />
    </div>
  );
}
