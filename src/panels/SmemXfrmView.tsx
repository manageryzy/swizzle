// Middle "act" of the SMEM swizzle story: a compact SVG that draws short
// arrows from logical cell centres to physical cell centres under the active
// Swizzle<B,M,S>. Sampling: one arrow every `stride` logical cells so the
// picture stays readable for large tiles. Arrows are animated by phase
// progress during ldmatrix / cp.async so the reader sees the transform sweep
// in lane order.

import { useMemo } from 'preact/hooks';
import {
  activePattern,
  activeSwizzleA,
  activeSwizzleB,
  currentAccesses,
  currentEpiloguePhase,
  currentProducerPhase,
  epiloguePhaseProgress,
  focusedOffset,
  inst,
  producerPhaseProgress,
  spec,
} from '../state';
import { apply } from '../swizzle';
import { accessArrival } from '../patterns';
import { tileDimsFor } from '../tile_dims';

const W = 240;
const H = 220;
const LEFT_PAD = 10;
const RIGHT_PAD = 10;
const TOP_PAD = 12;
const BOT_PAD = 12;
const ARROWS_PER_ROW = 6;
const ROWS_SAMPLED = 8;
const BYTES_PER_WORD = 4;
const LINE_BYTES = 128;

interface Props {
  operand: 'A' | 'B';
}

export function SmemXfrmView({ operand }: Props) {
  const s = spec.value;
  const i = inst.value;
  // Sweep progress: producer stream during load (tma.load / cp.async /
  // ldmatrix), epilogue stream during store (tma.store). The SVG stays idle
  // between those windows.
  const cProd = currentProducerPhase.value;
  const cEpi = currentEpiloguePhase.value;
  const pProg = producerPhaseProgress.value;
  const eProg = epiloguePhaseProgress.value;
  const accesses = currentAccesses.value;
  void activePattern.value;

  const major = operand === 'A' ? s.majorA : s.majorB;
  const dims = tileDimsFor(i, operand, major);
  const sw = (operand === 'A' ? activeSwizzleA : activeSwizzleB).value;

  const prodKind = cProd?.kind;
  const epiKind = cEpi?.kind;
  const isWarpRead = prodKind === 'ldmatrix' || prodKind === 'cp.async' || prodKind === 'tma.load';
  const isStore = epiKind === 'epilogue.tma.store';
  // Progress to drive the sweep: load sweeps forward in producer time,
  // store sweeps "back" via epilogue progress. If neither stream is active
  // (e.g. mid wgmma.step with no concurrent TMA), progress is 0.
  const progress = isWarpRead ? pProg : isStore ? eProg : 0;
  const arrival = useMemo(
    () => (isWarpRead ? accessArrival(accesses, sw) : null),
    [accesses, sw, isWarpRead],
  );

  // Sample grid.
  const rowStride = dims.rowStrideBytes;
  const cellsPerRow = Math.min(ARROWS_PER_ROW, Math.floor(rowStride / BYTES_PER_WORD));
  const rowsSampled = Math.min(ROWS_SAMPLED, dims.rows);

  const lx = (col: number) =>
    LEFT_PAD + (col / Math.max(1, cellsPerRow - 1)) * (W * 0.32 - LEFT_PAD);
  const ly = (row: number) =>
    TOP_PAD + (row / Math.max(1, rowsSampled - 1)) * (H - TOP_PAD - BOT_PAD);

  // Physical coordinates: one "line" (128 B) per row on the right. Map
  // physByte → (line, bankWord).
  const rx = (bankWord: number) =>
    W * 0.68 + (bankWord / 31) * (W - RIGHT_PAD - W * 0.68);
  const ry = (line: number) => {
    const lineCount = Math.max(1, Math.ceil(dims.tileBytes / LINE_BYTES));
    return TOP_PAD + (line / Math.max(1, Math.min(lineCount, rowsSampled) - 1)) * (H - TOP_PAD - BOT_PAD);
  };

  const hueFor = (row: number, col: number) => {
    const band = Math.floor(row / 2) % 8;
    return (band * 45 + col * 9) % 360;
  };

  const arrows = useMemo(() => {
    const out: {
      key: string;
      lx: number;
      ly: number;
      rx: number;
      ry: number;
      row: number;
      col: number;
      hue: number;
      fade: number;
    }[] = [];
    const waveCount = arrival ? Math.max(1, arrival.waveCount) : 1;
    const cur = progress * waveCount;
    for (let r = 0; r < rowsSampled; r++) {
      for (let c = 0; c < cellsPerRow; c++) {
        const logicalByte =
          Math.floor((r * rowStride * (dims.rows - 1)) / Math.max(1, rowsSampled - 1)) +
          c * Math.max(BYTES_PER_WORD, Math.floor(rowStride / cellsPerRow));
        if (logicalByte >= dims.tileBytes) continue;
        const phys = apply(sw, logicalByte);
        const line = Math.floor(phys / LINE_BYTES);
        const bankWord = (phys % LINE_BYTES) / BYTES_PER_WORD;
        // Per-lane animation: if this column corresponds to a known access,
        // derive a wave; otherwise just fade by overall progress.
        let fade = Math.min(1, progress * 1.6);
        if (arrival && accesses.length > 0) {
          // Map column c to a lane by modulo; this is purely cosmetic.
          const lane = (r * cellsPerRow + c) % 32;
          const w = arrival.wavePerLane.get(lane) ?? 0;
          fade = Math.max(0, Math.min(1, (cur - w) * 3));
        }
        out.push({
          key: `${r}-${c}`,
          lx: lx(c),
          ly: ly(r),
          rx: rx(bankWord),
          ry: ry(line),
          row: r,
          col: c,
          hue: hueFor(r, c),
          fade,
        });
      }
    }
    return out;
  }, [sw.B, sw.M, sw.S, dims.rowStrideBytes, dims.tileBytes, dims.rows, progress, isWarpRead, cellsPerRow, rowsSampled, accesses.length, arrival?.waveCount]);

  const focus = focusedOffset.value;

  return (
    <div class="smem-xfrm">
      <div class="smem-xfrm__head">
        <span class="smem-xfrm__kind">swizzle</span>
        <span class="smem-xfrm__meta">apply(o) = o ⊕ ((o &amp; yyy) &gt;&gt; {sw.S})</span>
      </div>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} class="smem-xfrm__svg">
        {/* logical source column */}
        <rect x={0} y={0} width={W * 0.32} height={H} class="smem-xfrm__col smem-xfrm__col--src" />
        <text x={W * 0.16} y={14} text-anchor="middle" class="smem-xfrm__collbl">logical</text>
        {/* physical destination column */}
        <rect x={W * 0.68} y={0} width={W * 0.32} height={H} class="smem-xfrm__col smem-xfrm__col--dst" />
        <text x={W * 0.68 + W * 0.16} y={14} text-anchor="middle" class="smem-xfrm__collbl">physical</text>
        {/* Middle arrow hint when no active animation. */}
        {arrows.every((a) => a.fade < 0.05) && (
          <>
            <text x={W / 2} y={H / 2 - 10} text-anchor="middle" class="smem-xfrm__idlelbl">
              o ⊕ (yyy{` `}&gt;&gt;{` `}S)
            </text>
            <text x={W / 2} y={H / 2 + 8} text-anchor="middle" class="smem-xfrm__idlelbl smem-xfrm__idlelbl--dim">
              active during ldmatrix / cp.async
            </text>
          </>
        )}
        {arrows.map((a) => (
          <g key={a.key} style={{ opacity: a.fade }}>
            <circle cx={a.lx} cy={a.ly} r={2.5} fill={`hsl(${a.hue},70%,55%)`} />
            <line
              x1={a.lx}
              y1={a.ly}
              x2={a.rx}
              y2={a.ry}
              stroke={`hsl(${a.hue},70%,55%)`}
              stroke-width={0.8}
              stroke-opacity={0.7}
            />
            <circle cx={a.rx} cy={a.ry} r={2.5} fill={`hsl(${a.hue},70%,55%)`} stroke="#000" stroke-width={0.3} />
          </g>
        ))}
        {/* focused-offset arrow, if any */}
        {focus != null && focus >= 0 && focus < dims.tileBytes && (
          (() => {
            const phys = apply(sw, focus);
            const logicalRow = Math.floor(focus / rowStride);
            const logicalCol = Math.floor((focus % rowStride) / BYTES_PER_WORD);
            const line = Math.floor(phys / LINE_BYTES);
            const bankWord = (phys % LINE_BYTES) / BYTES_PER_WORD;
            const rRow = Math.min(rowsSampled - 1, Math.floor((logicalRow / Math.max(1, dims.rows - 1)) * (rowsSampled - 1)));
            return (
              <g class="smem-xfrm__focus">
                <circle cx={lx(Math.min(cellsPerRow - 1, logicalCol))} cy={ly(rRow)} r={5} fill="none" stroke="#fff" stroke-width={1.5} />
                <line
                  x1={lx(Math.min(cellsPerRow - 1, logicalCol))}
                  y1={ly(rRow)}
                  x2={rx(bankWord)}
                  y2={ry(Math.min(rowsSampled - 1, line))}
                  stroke="#fff"
                  stroke-width={1.5}
                />
                <circle cx={rx(bankWord)} cy={ry(Math.min(rowsSampled - 1, line))} r={5} fill="none" stroke="#fff" stroke-width={1.5} />
              </g>
            );
          })()
        )}
      </svg>
    </div>
  );
}
