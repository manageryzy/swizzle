import { useEffect, useRef } from 'preact/hooks';
import {
  activePattern,
  activeSwizzleA,
  activeSwizzleB,
  currentAccesses,
  currentKStep,
  currentPhase,
  focusedOffset,
  inst,
  phaseProgress,
  spec,
} from '../state';
import { apply } from '../swizzle';
import { accessArrival, contiguousArrival, type Access } from '../patterns';
import { tileDimsFor } from '../tile_dims';
import type { Major } from '../instructions';

const MAX_ROWS = 64;
const MAX_COLS = 64;
const MIN_CELL = 3;
const MAX_CELL = 10;
const BANK_WORDS_PER_LINE = 32; // 32 banks × 4 B = 128 B per line
const BYTES_PER_WORD = 4;
const LINE_BYTES = BANK_WORDS_PER_LINE * BYTES_PER_WORD;

export interface SmemTileViewProps {
  operand: 'A' | 'B';
  view: 'logical' | 'physical';
  /** K slice index held by this stage (used as a per-stage hue offset so
   *  the reader can see stages hold *different* data). */
  sliceIdx?: number;
}

// Deterministic colour per (K-slice, row, cellCol). Same matrix cell keeps
// its colour across logical/physical views, but different slices use a
// distinct hue offset so the reader can see each stage holds different data.
function matrixColor(row: number, cellCol: number, sliceIdx: number): string {
  const band = Math.floor(row / 8) % 8;
  const hue = (band * 45 + sliceIdx * 37) % 360;
  const light = 26 + ((cellCol & 7) * 3);
  const sat = 55;
  return `hsl(${hue}, ${sat}%, ${light}%)`;
}

function laneColor(lane: number): string {
  return `hsl(${(lane * 360) / 32}, 85%, 65%)`;
}

function cellsOfAccess(a: Access, rowStrideBytes: number, colsLogical: number) {
  const cells: { row: number; col: number }[] = [];
  for (let b = 0; b < a.bytes; b += BYTES_PER_WORD) {
    const off = a.byteOffset + b;
    const row = Math.floor(off / rowStrideBytes);
    const col = Math.floor((off % rowStrideBytes) / BYTES_PER_WORD);
    if (col < colsLogical) cells.push({ row, col });
  }
  return cells;
}

export function SmemTileView({ operand, view, sliceIdx = 0 }: SmemTileViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const s = spec.value;
  const _p = activePattern.value;
  const i = inst.value;
  const phase = currentPhase.value;
  const progress = phaseProgress.value;
  const kStep = currentKStep.value;
  const accesses = currentAccesses.value;

  const major: Major = operand === 'A' ? s.majorA : s.majorB;
  const dims = tileDimsFor(i, operand, major);
  // Per-operand effective swizzle — the byte-level M shifts with element size
  // (fp16 → M=1, fp32 → M=2, fp8 → M=0), so A and B can disagree in mixed-
  // precision kernels.
  const sw = (operand === 'A' ? activeSwizzleA : activeSwizzleB).value;

  // Logical: matrix shape in 4-B cells.
  const logicalRows = Math.min(dims.rows, MAX_ROWS);
  const logicalCols = Math.min(dims.cols, MAX_COLS);
  // Physical: fixed 32-bank-word width; height = number of 128-B lines.
  const physLines = Math.min(
    Math.ceil(dims.tileBytes / LINE_BYTES),
    MAX_ROWS,
  );
  const physCols = BANK_WORDS_PER_LINE;

  const rows = view === 'logical' ? logicalRows : physLines;
  const cols = view === 'logical' ? logicalCols : physCols;
  const CELL = Math.max(MIN_CELL, Math.min(MAX_CELL, Math.floor(260 / Math.max(rows, cols))));
  const W = cols * CELL;
  const H = rows * CELL;

  const isPhase = phase?.kind;
  const isMmaStep = isPhase === 'wgmma.step' || isPhase === 'tcgen05.mma.step';
  const isLoad = isPhase === 'tma.load' || isPhase === 'cp.async';
  const isEpiFill = isPhase === 'epilogue.stg_smem';
  const isEpiDrain = isPhase === 'epilogue.tma.store';
  const isWarpRead = isPhase === 'ldmatrix';

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

    // Paint cells. Both views use the same `matrixColor(logicalRow, logicalCol, sliceIdx)`
    // so the reader can follow a single matrix cell between the two canvases.
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        let logicalRow: number, logicalCol: number, inTile = true;
        if (view === 'logical') {
          logicalRow = r;
          logicalCol = c;
          if (r * dims.rowStrideBytes + c * BYTES_PER_WORD >= dims.tileBytes) inTile = false;
        } else {
          // Physical: (line, bank_word) → physical byte → unswizzle → matrix cell.
          const physByte = r * LINE_BYTES + c * BYTES_PER_WORD;
          if (physByte >= dims.tileBytes) {
            inTile = false;
            logicalRow = logicalCol = 0;
          } else {
            // Swizzle is involutive, so the inverse is apply(sw, physByte).
            const logicalByte = apply(sw, physByte);
            logicalRow = Math.floor(logicalByte / dims.rowStrideBytes);
            logicalCol = Math.floor((logicalByte % dims.rowStrideBytes) / BYTES_PER_WORD);
          }
        }
        ctx.fillStyle = inTile ? matrixColor(logicalRow, logicalCol, sliceIdx) : '#0d1017';
        ctx.fillRect(c * CELL, r * CELL, CELL - 1, CELL - 1);
      }
    }

    // 128-B-line separators on the PHYSICAL view (one grid-line every row).
    if (view === 'physical' && CELL >= 5) {
      ctx.strokeStyle = 'rgba(255,255,255,0.12)';
      ctx.lineWidth = 1;
      for (let r = 1; r < rows; r++) {
        ctx.beginPath();
        ctx.moveTo(0, r * CELL);
        ctx.lineTo(W, r * CELL);
        ctx.stroke();
      }
    }

    // Lane access overlay. Both views project the per-lane byte offset: on
    // LOGICAL → (row, col) of the access; on PHYSICAL → (line, bank_word) of
    // the swizzled offset.
    if (isWarpRead || isPhase === 'cp.async') {
      ctx.lineWidth = Math.max(1, Math.floor(CELL / 6));
      const sched = isWarpRead ? accessArrival(accesses, sw) : contiguousArrival(accesses);
      const waveCount = Math.max(1, sched.waveCount);
      const cur = progress * waveCount;
      for (const a of accesses) {
        const w = sched.wavePerLane.get(a.laneId) ?? 0;
        const fade = Math.max(0, Math.min(1, (cur - w) * 3));
        if (fade <= 0) continue;
        ctx.globalAlpha = fade;
        ctx.strokeStyle = laneColor(a.laneId);
        for (const { row, col } of cellsOfAccess(a, dims.rowStrideBytes, logicalCols)) {
          let drawR: number, drawC: number;
          if (view === 'logical') {
            drawR = row;
            drawC = col;
          } else {
            const physByte = apply(sw, row * dims.rowStrideBytes + col * BYTES_PER_WORD);
            drawR = Math.floor(physByte / LINE_BYTES);
            drawC = Math.floor((physByte % LINE_BYTES) / BYTES_PER_WORD);
          }
          if (drawR >= 0 && drawR < rows && drawC >= 0 && drawC < cols) {
            ctx.strokeRect(drawC * CELL + 1, drawR * CELL + 1, CELL - 3, CELL - 3);
          }
        }
      }
      ctx.globalAlpha = 1;
    }

    // mma atom pulse (on both views).
    if (isMmaStep && kStep) {
      const pulse = 0.35 + 0.25 * Math.sin(progress * Math.PI);
      ctx.strokeStyle = `rgba(255, 216, 120, ${pulse})`;
      ctx.lineWidth = 2;
      ctx.strokeRect(1, 1, W - 2, H - 2);
    }

    // TMA dim-to-bright ramp.
    if (isLoad) {
      const tmaDim = 0.35 + 0.45 * Math.min(1, progress * 1.1);
      if (tmaDim < 1) {
        ctx.fillStyle = `rgba(13, 16, 23, ${1 - tmaDim})`;
        ctx.fillRect(0, 0, W, H);
      }
    }

    // Epilogue fill / drain (LOGICAL view: L→R fill; PHYSICAL view: drain).
    if (isEpiFill) {
      const fillCols = Math.ceil(cols * progress);
      ctx.fillStyle = 'rgba(255, 200, 120, 0.30)';
      ctx.fillRect(0, 0, fillCols * CELL, H);
    }
    if (isEpiDrain) {
      const drainCols = Math.ceil(cols * progress);
      ctx.fillStyle = 'rgba(13, 16, 23, 0.50)';
      ctx.fillRect((cols - drainCols) * CELL, 0, drainCols * CELL, H);
    }
  }, [sw.B, sw.M, sw.S, s, _p.id, accesses, isMmaStep, isLoad, isWarpRead, isEpiFill, isEpiDrain, isPhase, progress, kStep?.k, kStep?.total, dims.rows, dims.cols, dims.rowStrideBytes, dims.tileBytes, CELL, view, rows, cols, W, H, operand, major, logicalCols, sliceIdx]);

  function onMove(e: MouseEvent) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const col = Math.floor(x / CELL);
    const row = Math.floor(y / CELL);
    if (row < 0 || col < 0 || row >= rows || col >= cols) return;
    const byteOff =
      view === 'logical'
        ? row * dims.rowStrideBytes + col * BYTES_PER_WORD
        : row * LINE_BYTES + col * BYTES_PER_WORD; // physical offset
    // Always record a LOGICAL byte offset for the bitfield panel.
    focusedOffset.value =
      view === 'logical' ? byteOff : apply(sw, byteOff);
  }

  const subtitle = view === 'logical'
    ? `matrix ${logicalRows}×${logicalCols} cells (${dims.dtypeLabel}, ${major}-major)`
    : `SRAM ${physLines} × 128 B lines · 32 banks/line`;

  return (
    <div class="smem-tileview">
      <div class="smem-tileview__head">
        <span class="smem-tileview__kind">{view}</span>
        <span class="smem-tileview__dims">{subtitle}</span>
      </div>
      <canvas ref={canvasRef} onMouseMove={onMove} />
    </div>
  );
}
