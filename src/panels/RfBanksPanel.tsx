import { useEffect, useRef, useState } from 'preact/hooks';
import {
  BANKS_PER_SUBPART,
  BYTES_PER_ENTRY,
  SUBPARTITIONS,
  fragmentRegIds,
  readCycles,
  regToPhysical,
} from '../rf';
import { clayoutOf } from '../cute_mma_layouts';
import { currentPhase, inst, phaseProgress } from '../state';

// Geometry: banks are columns, entries are rows. Each subpartition = 2 banks
// drawn adjacent; a small gap separates subpartitions. The active warp's 2
// banks get a bright border.
const VIS_ENTRIES = 32; // first 32 of 256
const CELL_W = 14;
const CELL_H = 10;
const SUBPART_GAP = 12;
const BANK_GAP = 2;

function fragColor(kind: 'free' | 'frag' | 'frag-read' | 'frag-write' | 'inactive'): string {
  switch (kind) {
    case 'free': return '#1a1f2a';
    case 'inactive': return '#14161d';
    case 'frag': return '#4a6f8a';
    case 'frag-read': return '#7ec699';
    case 'frag-write': return '#e0cf7a';
  }
}

export function RfBanksPanel() {
  const i = inst.value;
  const phase = currentPhase.value;
  const progress = phaseProgress.value;
  const [subpart, setSubpart] = useState(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const clayout = clayoutOf(i.id);
  const regsPerLane = clayout?.valuesPerThread ?? (i.accIn === 'tmem' ? 0 : 4);
  const fragRegs = fragmentRegIds(regsPerLane);
  const cycles = readCycles(fragRegs);

  // For each frag reg, the collector cycle = its position in its bank's queue
  // (even/odd split). Regs are served one bank at a time by the single read port.
  const cycleOfReg = new Map<number, number>();
  const posInBank = [0, 0];
  for (const r of fragRegs) {
    const parity = r & 1;
    cycleOfReg.set(r, posInBank[parity]++);
  }

  const mode: 'read' | 'write' | 'hold' = (() => {
    if (!phase) return 'hold';
    if (phase.kind === 'ldmatrix' || phase.kind === 'tcgen05.ld') return 'write';
    if (phase.kind === 'wgmma.step' || phase.kind === 'tcgen05.mma.step') {
      return progress < 0.5 ? 'read' : 'write';
    }
    if (phase.kind === 'epilogue.stg_smem' || phase.kind === 'epilogue.tma.store' || phase.kind === 'tcgen05.st') return 'read';
    return 'hold';
  })();
  const subProgress =
    phase?.kind === 'wgmma.step' || phase?.kind === 'tcgen05.mma.step'
      ? (mode === 'read' ? progress * 2 : (progress - 0.5) * 2)
      : progress;
  const activeCycle = Math.floor(subProgress * cycles);

  // Canvas dims.
  const bankW = CELL_W;
  const subpartW = BANKS_PER_SUBPART * bankW + (BANKS_PER_SUBPART - 1) * BANK_GAP;
  const totalW = SUBPARTITIONS * subpartW + (SUBPARTITIONS - 1) * SUBPART_GAP;
  const totalH = VIS_ENTRIES * CELL_H;
  const HEADER_H = 16;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const W = totalW;
    const H = totalH + HEADER_H;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = `${W}px`;
    canvas.style.height = `${H}px`;
    const ctx = canvas.getContext('2d')!;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    // Precompute which (bank, entry) slots host the fragment and at what
    // cycle each entry becomes hot.
    const fragByBank = new Map<number, Map<number, number>>();
    for (const r of fragRegs) {
      const p = regToPhysical(subpart, r);
      const map = fragByBank.get(p.bank) ?? new Map<number, number>();
      map.set(p.entry, cycleOfReg.get(r) ?? 0);
      fragByBank.set(p.bank, map);
    }

    // Bank header row.
    ctx.font = '9px ui-monospace, monospace';
    for (let sp = 0; sp < SUBPARTITIONS; sp++) {
      const spX = sp * (subpartW + SUBPART_GAP);
      const activeSp = sp === subpart;
      ctx.fillStyle = activeSp ? '#e6e1cf' : '#5a6378';
      ctx.fillText(`subp ${sp}`, spX + 2, 9);
      for (let b = 0; b < BANKS_PER_SUBPART; b++) {
        const bankGlobal = sp * BANKS_PER_SUBPART + b;
        const bx = spX + b * (bankW + BANK_GAP);
        ctx.fillStyle = activeSp ? '#7e9cd8' : '#5a6378';
        ctx.fillText(`B${bankGlobal}`, bx + 1, 20);
      }
    }

    // Entry grid.
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
        // Bank outline — bright for active subpartition.
        if (activeSp) {
          ctx.strokeStyle = '#7e9cd8';
          ctx.lineWidth = 1;
          ctx.strokeRect(bx - 0.5, top - 0.5, CELL_W, VIS_ENTRIES * CELL_H);
        }
      }
    }

    // Collector-port label: "1 read port per bank · X cycles to read frag".
    ctx.fillStyle = cycles > 1 ? '#e07878' : '#7ec699';
    ctx.font = '9px ui-monospace, monospace';
    ctx.fillText(
      `cycle ${activeCycle + 1}/${cycles} · 1 rd port/bank`,
      0,
      top + VIS_ENTRIES * CELL_H + 11,
    );
  }, [subpart, fragRegs.join(','), mode, activeCycle, cycles, totalW]);

  return (
    <div class="panel rf">
      <h3>
        LRF (register file){' '}
        <small>
          — 8 banks × 256 entries × 128 B · 4 subpartitions · 2 banks/warp · 1 rd port/bank
        </small>
      </h3>
      <div class="rf__canvas-wrap">
        <canvas ref={canvasRef} />
      </div>
      <p class="panel__note">
        active warp in <code>subp {subpart}</code>, banks{' '}
        <code>B{subpart * 2}, B{subpart * 2 + 1}</code>
        {regsPerLane > 0 && (
          <>
            {' '}· fragment: <code>{regsPerLane} regs/lane</code> ({regsPerLane * BYTES_PER_ENTRY} B/warp) · collector <span class={cycles > 1 ? 'rf__stall' : ''}>{cycles}c</span>
          </>
        )}
        <button
          class="rf__subp-btn"
          onClick={() => setSubpart((s) => (s + 1) % SUBPARTITIONS)}
        >
          next subp
        </button>
      </p>
      <div class="rf__legend">
        <span class="rf__swatch" style={{ background: fragColor('free') }} /> free
        <span class="rf__swatch" style={{ background: fragColor('frag') }} /> held by frag
        <span class="rf__swatch" style={{ background: fragColor('frag-read') }} /> reading
        <span class="rf__swatch" style={{ background: fragColor('frag-write') }} /> writing
        <span class="rf__swatch" style={{ background: fragColor('inactive') }} /> inactive (other warp)
      </div>
    </div>
  );
}
