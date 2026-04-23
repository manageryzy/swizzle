// SimtRegPanel — merged view of (1) which thread in the warpgroup owns which
// element of the C-tile and (2) where that thread's fragment lives in the RF
// banks. Click a C-tile cell to set `laneSel` (and `warpSel` to the warp that
// owns it); the RF grid on the right highlights the fragment regs that
// belong to that lane, and the collector strip at the bottom shows the
// port-serialised read cycles needed to deliver those regs to the MMA.

import { useEffect, useMemo, useRef } from 'preact/hooks';
import {
  blkMMult,
  blkNMult,
  currentConsumerPhase,
  currentEpiloguePhase,
  currentProducerPhase,
  epiloguePhaseProgress,
  inst,
  phaseProgress,
  producerPhaseProgress,
  spec,
  summary,
  laneSel,
  warpSel,
  warpsInGroup,
  world,
} from '../state';
import type { WarpState } from '../simulation';
import { clayoutOf, ownershipMap } from '../cute_mma_layouts';
import { layoutAt } from '../cute_layout';
import {
  BANKS_PER_SUBPART,
  BYTES_PER_ENTRY,
  SUBPARTITIONS,
  fragmentRegIds,
  readCycles,
  regToPhysical,
} from '../rf';
import { TruthFooter } from './TruthFooter';

function threadColor(t: number, totalThreads: number): string {
  return `hsl(${(t * 360) / totalThreads}, 75%, 55%)`;
}

// Fallback owner map for instructions without a ported CLayout.
function ownerFallback(m: number, n: number, threadsPerMma: number): number {
  const quad = (m % 8) >> 1;
  const warp = (m >> 3) & 3;
  const col = n & 7;
  return (warp * 32 + quad * 8 + col) % threadsPerMma;
}

const VIS_ENTRIES = 32;
const CELL_W = 14;
const CELL_H = 10;
const SUBPART_GAP = 12;
const BANK_GAP = 2;

function fragColor(kind: 'free' | 'frag' | 'frag-read' | 'frag-write' | 'inactive' | 'frag-laneonly'): string {
  switch (kind) {
    case 'free': return '#1a1f2a';
    case 'inactive': return '#14161d';
    case 'frag': return '#4a6f8a';
    case 'frag-read': return '#7ec699';
    case 'frag-write': return '#e0cf7a';
    case 'frag-laneonly': return '#e09a5a';
  }
}

// Phase 6 — per-warp role chip colour. Mirrors the plan §L2 palette.
function roleColor(role: WarpState['role']): string {
  switch (role) {
    case 'producer': return '#7ec699';   // green
    case 'consumer': return '#e0cf7a';   // yellow
    case 'epilogue': return '#7e9cd8';   // blue
    case 'idle': return '#3a4050';       // grey
  }
}

export function SimtRegPanel() {
  const i = inst.value;
  const s = spec.value;
  // Per-stream phases + progress. Different animations track different
  // streams: ldmatrix (producer sub-phase on RS/coupled) uses producer
  // progress, mma.step uses consumer progress, epilogue uses epilogue.
  const cProd = currentProducerPhase.value;
  const cCons = currentConsumerPhase.value;
  const cEpi = currentEpiloguePhase.value;
  const pProg = producerPhaseProgress.value;
  const cProg = phaseProgress.value;
  const eProg = epiloguePhaseProgress.value;
  void cProd;
  const w = world.value;
  const consAtom = w.consumerAtom;
  const kStep = consAtom
    ? { k: consAtom.kStep, total: summary.value.consumerItersTotal }
    : null;
  const owners = canvasRefs();

  const threadsPerMma = i.family === 'wmma' || i.family === 'mma' ? 32 : 128;
  const warps = warpsInGroup.value;
  const warp = Math.min(warpSel.value, Math.max(0, warps - 1));
  const lane = laneSel.value;
  const isWarpGroup = i.family === 'wgmma' || i.family === 'tcgen05' || i.family === 'tcgen05.block_scaled';
  // Phase 6 — per-warp role chip strip. Shown for .ws variant so users see
  // warp 0 acting as producer while warps 1..3 consume. For non-.ws we
  // still expose the 4 warps (all sharing a role) so the UI stays stable;
  // but the strip is hidden for single-warp families (mma, wmma).
  const sumV = summary.value;
  const isWs = sumV.extras.warpSpecialized;
  const warpStates = world.value.warps;
  const showRoleStrip = warpStates.length > 1;

  const clayout = clayoutOf(i.id);
  const realOwner = clayout ? ownershipMap(clayout) : null;
  const ownerOf = (m: number, n: number) =>
    realOwner ? realOwner(m, n) : ownerFallback(m, n, threadsPerMma);

  // --- Left: C-tile fill order (from RmemPanel) -----------------------------
  const fillOrder = useMemo(() => {
    const out: Array<{ m: number; n: number }> = [];
    if (clayout) {
      for (let tid = 0; tid < clayout.threads; tid++) {
        for (let vid = 0; vid < clayout.valuesPerThread; vid++) {
          const idx = tid + vid * clayout.threads;
          const off = layoutAt(clayout.shape, clayout.stride, idx);
          const m = off % clayout.M;
          const n = Math.floor(off / clayout.M);
          out.push({ m, n });
        }
      }
    } else {
      for (let t = 0; t < threadsPerMma; t++) {
        for (let m = 0; m < i.M; m++)
          for (let n = 0; n < i.N; n++)
            if (ownerFallback(m, n, threadsPerMma) === t) out.push({ m, n });
      }
    }
    return out;
  }, [clayout?.label, threadsPerMma, i.M, i.N, i.id]);

  const fill = (() => {
    if (cCons?.kind === 'wgmma.step' || cCons?.kind === 'tcgen05.mma.step') {
      if (!kStep) return 0;
      return Math.min(1, (kStep.k + cProg) / kStep.total);
    }
    const after = cEpi && (cEpi.kind === 'tcgen05.ld' || cEpi.kind === 'epilogue.tma.store');
    return after ? 1 : 0;
  })();

  // Show the full tile when we can. Scale the cell down so nothing is
  // silently clipped — the C-tile ownership is the whole point of this
  // panel, so honest truncation beats hiding half the columns.
  const rows = Math.min(i.M, 128);
  const cols = Math.min(i.N, 256);
  const cell = Math.max(3, Math.min(10, Math.floor(420 / Math.max(rows, cols))));
  const rmemW = cols * cell;
  const rmemH = rows * cell;

  // Selected cell = a canonical (m,n) for the currently-selected (warp,lane).
  // We search fillOrder for the first cell whose owner matches.
  const selectedThread = lane == null ? null : warp * 32 + lane;
  const selectedCell = useMemo(() => {
    if (selectedThread == null) return null;
    for (const { m, n } of fillOrder) {
      if (ownerOf(m, n) === selectedThread) return { m, n };
    }
    return null;
  }, [selectedThread, fillOrder, i.id]);

  // Phase 5: per-atom accumulation opacity. Each cell of the C-tile belongs
  // to an atom (atomM, atomN) within the blkMMult × blkNMult atom grid; the
  // opacity reflects world.cTile.accumulated[atomM][atomN] /
  // summary.maxAccumulatedPerAtom. In the default single-atom case
  // (blkMMult=blkNMult=1) this is just a global fill. The panel always draws
  // inst.M × inst.N cells (one atom face); we still honour per-atom opacity
  // in case the panel is later scaled to show the full blk tile.
  const atomMSpan = Math.max(1, i.M);
  const atomNSpan = Math.max(1, i.N);
  const bm = blkMMult.value;
  const bn = blkNMult.value;
  const maxAcc = summary.value.maxAccumulatedPerAtom;
  const cAcc = world.value.cTile?.accumulated;
  const accOpacity = (m: number, n: number): number => {
    if (maxAcc <= 0 || !cAcc) return 0;
    const am = Math.min(bm - 1, Math.floor(m / atomMSpan));
    const an = Math.min(bn - 1, Math.floor(n / atomNSpan));
    const a = cAcc[am]?.[an] ?? 0;
    return Math.min(1, a / maxAcc);
  };

  useEffect(() => {
    const canvas = owners.rmemRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rmemW * dpr;
    canvas.height = rmemH * dpr;
    canvas.style.width = `${rmemW}px`;
    canvas.style.height = `${rmemH}px`;
    const ctx = canvas.getContext('2d')!;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, rmemW, rmemH);
    ctx.fillStyle = '#0d1017';
    ctx.fillRect(0, 0, rmemW, rmemH);

    const filled = Math.floor(fill * fillOrder.length);
    for (let idx = 0; idx < filled && idx < fillOrder.length; idx++) {
      const { m, n } = fillOrder[idx];
      if (m >= rows || n >= cols) continue;
      // Per-atom opacity: accumulation fraction fades the owner colour in.
      const opa = accOpacity(m, n);
      const base = threadColor(ownerOf(m, n), threadsPerMma);
      ctx.globalAlpha = Math.max(0.15, opa); // keep a floor so layout is legible
      ctx.fillStyle = base;
      ctx.fillRect(n * cell, m * cell, cell - 1, cell - 1);
      ctx.globalAlpha = 1;
    }

    // Highlight cells owned by the selected thread.
    if (selectedThread != null) {
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.2;
      for (const { m, n } of fillOrder) {
        if (m >= rows || n >= cols) continue;
        if (ownerOf(m, n) !== selectedThread) continue;
        ctx.strokeRect(n * cell + 0.5, m * cell + 0.5, cell - 2, cell - 2);
      }
    }

    // Pen-tip at the currently-writing cell.
    if (fill > 0 && fill < 1 && filled < fillOrder.length) {
      const { m, n } = fillOrder[filled] ?? { m: 0, n: 0 };
      if (m < rows && n < cols) {
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(n * cell, m * cell, cell - 1, cell - 1);
      }
    }
  }, [rows, cols, cell, threadsPerMma, i.id, fill, fillOrder, selectedThread, rmemW, rmemH, maxAcc, bm, bn, JSON.stringify(cAcc ?? null)]);

  function onRmemClick(ev: MouseEvent) {
    const c = owners.rmemRef.current;
    if (!c) return;
    const rect = c.getBoundingClientRect();
    const n = Math.floor((ev.clientX - rect.left) / cell);
    const m = Math.floor((ev.clientY - rect.top) / cell);
    if (m < 0 || m >= rows || n < 0 || n >= cols) return;
    const tid = ownerOf(m, n);
    const newWarp = Math.floor(tid / 32);
    const newLane = tid % 32;
    warpSel.value = Math.min(newWarp, Math.max(0, warps - 1));
    laneSel.value = newLane;
  }

  // --- Right: RF bank grid (from RfBanksPanel) ------------------------------
  const regsPerLane = clayout?.valuesPerThread ?? (i.accIn === 'tmem' ? 0 : 4);
  const fragRegs = fragmentRegIds(regsPerLane);
  const cycles = readCycles(fragRegs);

  // For each reg, its collector-cycle = position in its bank queue.
  const cycleOfReg = useMemo(() => {
    const m = new Map<number, number>();
    const posInBank = [0, 0];
    for (const r of fragRegs) {
      const parity = r & 1;
      m.set(r, posInBank[parity]++);
    }
    return m;
  }, [fragRegs.join(',')]);

  // Mode + its progress — each mode pulls progress from the stream that owns
  // the phase (producer for ldmatrix, consumer for mma.step, epilogue for
  // stg_smem/tma.store/tcgen05.ld/tcgen05.st). This keeps the collector
  // strip in lockstep with whichever stream is actually active.
  const prodKind = cProd?.kind;
  const consKind = cCons?.kind;
  const epiKind = cEpi?.kind;
  const { mode, modeProg } = ((): { mode: 'read' | 'write' | 'hold'; modeProg: number } => {
    if (prodKind === 'ldmatrix') return { mode: 'write', modeProg: pProg };
    if (epiKind === 'tcgen05.ld') return { mode: 'write', modeProg: eProg };
    if (consKind === 'wgmma.step' || consKind === 'tcgen05.mma.step') {
      return cProg < 0.5
        ? { mode: 'read', modeProg: cProg }
        : { mode: 'write', modeProg: cProg };
    }
    if (epiKind === 'epilogue.stg_smem' || epiKind === 'epilogue.tma.store' || epiKind === 'tcgen05.st') {
      return { mode: 'read', modeProg: eProg };
    }
    return { mode: 'hold', modeProg: 0 };
  })();
  const subProgress =
    consKind === 'wgmma.step' || consKind === 'tcgen05.mma.step'
      ? (mode === 'read' ? modeProg * 2 : (modeProg - 0.5) * 2)
      : modeProg;
  const activeCycle = Math.floor(subProgress * cycles);

  const bankW = CELL_W;
  const subpartW = BANKS_PER_SUBPART * bankW + (BANKS_PER_SUBPART - 1) * BANK_GAP;
  const totalW = SUBPARTITIONS * subpartW + (SUBPARTITIONS - 1) * SUBPART_GAP;
  const totalH = VIS_ENTRIES * CELL_H;
  const HEADER_H = 16;
  const STRIP_H = 22; // room for the collector-port cycle strip below the grid

  useEffect(() => {
    const canvas = owners.rfRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const W = totalW;
    const H = totalH + HEADER_H + STRIP_H;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = `${W}px`;
    canvas.style.height = `${H}px`;
    const ctx = canvas.getContext('2d')!;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    // Fragment → (bank, entry) mapping for the active subpartition.
    const subpart = warp % SUBPARTITIONS;
    const fragByBank = new Map<number, Map<number, number>>();
    for (const r of fragRegs) {
      const p = regToPhysical(subpart, r);
      const map = fragByBank.get(p.bank) ?? new Map<number, number>();
      map.set(p.entry, cycleOfReg.get(r) ?? 0);
      fragByBank.set(p.bank, map);
    }

    ctx.font = '9px "JetBrains Mono", Menlo, monospace';
    for (let sp = 0; sp < SUBPARTITIONS; sp++) {
      const spX = sp * (subpartW + SUBPART_GAP);
      const activeSp = sp === subpart;
      ctx.fillStyle = activeSp ? '#e6e1cf' : '#5a6378';
      ctx.fillText(`W${sp}`, spX + 2, 9);
      for (let b = 0; b < BANKS_PER_SUBPART; b++) {
        const bankGlobal = sp * BANKS_PER_SUBPART + b;
        const bx = spX + b * (bankW + BANK_GAP);
        ctx.fillStyle = activeSp ? '#7e9cd8' : '#5a6378';
        ctx.fillText(`B${bankGlobal}`, bx + 1, 20);
      }
    }

    const top = HEADER_H;
    for (let sp = 0; sp < SUBPARTITIONS; sp++) {
      const activeSp = sp === subpart;
      const spX = sp * (subpartW + SUBPART_GAP);
      for (let b = 0; b < BANKS_PER_SUBPART; b++) {
        const bankGlobal = sp * BANKS_PER_SUBPART + b;
        const bx = spX + b * (bankW + BANK_GAP);
        for (let e = 0; e < VIS_ENTRIES; e++) {
          const y = top + e * CELL_H;
          const entryCycle = fragByBank.get(bankGlobal)?.get(e);
          const inFrag = entryCycle !== undefined;
          let color: string;
          if (!activeSp) color = fragColor('inactive');
          else if (inFrag) {
            const hot = entryCycle! <= activeCycle;
            if (mode === 'read' && hot) color = fragColor('frag-read');
            else if (mode === 'write' && hot) color = fragColor('frag-write');
            else color = fragColor('frag');
          } else color = fragColor('free');
          ctx.fillStyle = color;
          ctx.fillRect(bx, y, CELL_W - 1, CELL_H - 1);
        }
        if (activeSp) {
          ctx.strokeStyle = '#7e9cd8';
          ctx.lineWidth = 1;
          ctx.strokeRect(bx - 0.5, top - 0.5, CELL_W, VIS_ENTRIES * CELL_H);
        }
      }
    }

    // Collector-port cycle strip at the bottom of the RF canvas.
    const stripY = top + VIS_ENTRIES * CELL_H + 4;
    ctx.fillStyle = '#5a6378';
    ctx.fillText(`collector`, 0, stripY + 8);
    for (let c = 0; c < cycles; c++) {
      const x = 60 + c * 14;
      const active = c <= activeCycle;
      ctx.fillStyle = active ? (mode === 'read' ? '#7ec699' : '#e0cf7a') : '#2a2f3a';
      ctx.fillRect(x, stripY, 10, 10);
      ctx.strokeStyle = c === activeCycle ? '#fff' : 'transparent';
      ctx.lineWidth = 1;
      ctx.strokeRect(x - 0.5, stripY - 0.5, 11, 11);
    }
  }, [warp, fragRegs.join(','), mode, activeCycle, cycles, totalW]);

  return (
    <div class="panel simtreg">
      <h3>
        SIMT registers{' '}
        <small>
          — C-tile ownership ({i.M}×{i.N}) ↔ RF banks (8 × 256 entries)
        </small>
      </h3>

      <div class="simtreg__head">
        <span class="simtreg__meta">
          {isWarpGroup ? `warpgroup · ${warps} warps × 32 lanes` : 'single warp · 32 lanes'} ·{' '}
          A source: <code class={`pill--src ${s.aSource === 'tmem' ? 'pill--tmem' : ''}`}>{s.aSource.toUpperCase()}</code>
        </span>
        <span class="simtreg__selmeta">
          {selectedThread != null ? (
            <>
              selected: warp <code>W{warp}</code> · lane <code>L{lane}</code> · thread{' '}
              <code>T{selectedThread}</code>
              {selectedCell && (
                <>
                  {' '}owns <code>C[{selectedCell.m}, {selectedCell.n}]</code>
                </>
              )}
            </>
          ) : (
            <span class="simtreg__muted">click a C-tile cell (or key l) to pick a lane</span>
          )}
        </span>
      </div>

      {/* Phase 6 — per-warp role chip strip (plan §L3). Shows each warp's
          current role with a coloured chip. For .ws this visualises warp 0
          producing while warps 1..3 consume; for non-.ws all 4 chips share
          the same role. */}
      {showRoleStrip && (
        <div class="simtreg__warp-strip" style={{
          display: 'flex',
          gap: '6px',
          margin: '4px 0 8px 0',
          alignItems: 'center',
          fontFamily: '"JetBrains Mono", Menlo, monospace',
          fontSize: '10px',
        }}>
          <span class="simtreg__muted" style={{ marginRight: '4px' }}>
            {isWs ? 'warp roles (.ws)' : 'warp roles'}:
          </span>
          {warpStates.map((ws) => (
            <span
              key={`wrole-${ws.warpIdx}`}
              title={`W${ws.warpIdx}: ${ws.role}`}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '4px',
                padding: '2px 6px',
                borderRadius: '3px',
                background: roleColor(ws.role),
                color: '#0b0e14',
                fontWeight: 600,
              }}
            >
              W{ws.warpIdx}
              <span style={{ opacity: 0.75, fontWeight: 400 }}>{ws.role}</span>
            </span>
          ))}
        </div>
      )}

      <div class="simtreg__grid">
        <div class="simtreg__col">
          <div class="simtreg__collbl">C-tile · colour = owning thread</div>
          <canvas ref={owners.rmemRef} onClick={onRmemClick} style={{ cursor: 'pointer' }} />
          {rows < i.M || cols < i.N ? (
            <p class="panel__note panel__note--dim">
              (display clipped to 64×64; inst is {i.M}×{i.N})
            </p>
          ) : null}
        </div>
        <div class="simtreg__col">
          <div class="simtreg__collbl">
            RF banks · subp {warp % SUBPARTITIONS} active ·{' '}
            <span class={cycles > 1 ? 'rf__stall' : ''}>{cycles} collector cycle{cycles > 1 ? 's' : ''}</span>
          </div>
          <div class="rf__canvas-wrap">
            <canvas ref={owners.rfRef} />
          </div>
          <div class="rf__legend">
            <span class="rf__swatch" style={{ background: fragColor('free') }} /> free
            <span class="rf__swatch" style={{ background: fragColor('frag') }} /> held
            <span class="rf__swatch" style={{ background: fragColor('frag-read') }} /> reading
            <span class="rf__swatch" style={{ background: fragColor('frag-write') }} /> writing
            <span class="rf__swatch" style={{ background: fragColor('inactive') }} /> inactive
          </div>
        </div>
      </div>

      <p class={`panel__note ${realOwner ? '' : 'panel__note--dim'}`}>
        fragment layout:{' '}
        {clayout ? (
          <>
            <code>{clayout.label}</code> (exact — from <code>{clayout.sourceFile}</code>) ·{' '}
            {regsPerLane} regs/lane ({regsPerLane * BYTES_PER_ENTRY} B/warp)
          </>
        ) : i.accIn === 'tmem' ? (
          <>
            tcgen05 accumulator lives in <code>TMEM</code>; this panel shows an
            approximate thread mapping so the epilogue <code>tcgen05.ld → .reg</code>{' '}
            step has a target.
          </>
        ) : (
          <>approximate (no ported CLayout for {i.id})</>
        )}
      </p>

      <TruthFooter
        verified={clayout != null}
        models="thread → C-tile mapping (ported from CUTE CLayout); 8-bank × 4-subpart RF topology; 1 read-port-per-bank collector cycles; per-warp role strip for .ws and multi-warp families from world.warps; C-tile ownership opacity follows world.cTile.accumulated."
        schematic="fragment tracking per lane (world.warps[].fragment is currently a stub); write-port contention; RAW/WAW hazards; sub-warp scheduling."
        cite="rf.ts · regToPhysical / readCycles; cute_mma_layouts.ts · CLayout"
      />
    </div>
  );
}

// Named helper so useRef wrappers don't pollute the component body.
function canvasRefs() {
  return {
    rmemRef: useRef<HTMLCanvasElement>(null),
    rfRef: useRef<HTMLCanvasElement>(null),
  };
}
