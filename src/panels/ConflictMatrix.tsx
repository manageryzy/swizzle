// SIMT bank-conflict matrix: 32 lanes × 32 banks SVG grid showing which lane
// lands in which bank under the active pattern × swizzle combination. Below
// the grid is a replay-cycle strip: N serialized waves for an N-way conflict.
//
// Honesty disclaimer: we model a single warp of 32 lanes accessing 4-B words.
// The underlying `accessArrival` already derives the per-lane wave index —
// we just make it visible here.

import { useMemo } from 'preact/hooks';
import {
  activePattern,
  activeSwizzle,
  activeTileDims,
  currentAccesses,
  currentConflicts,
  maxConflict,
  laneSel,
  bankSel,
  cycleSel,
  spec,
  world,
} from '../state';
import { apply, bankOfByte, effectiveSwizzle } from '../swizzle';
import { accessArrival } from '../patterns';
import { TruthFooter } from './TruthFooter';

const CELL = 14;
const HEAD_H = 18;
const PAD_L = 28;
const PAD_T = 8;

type CellData = {
  lane: number;
  bank: number;
  byteOffset: number;
  physOffset: number;
  word: number; // which 4-B word within the lane's access
};

function buildCells(accesses: { laneId: number; byteOffset: number; bytes: number }[], sw: { B: number; M: number; S: number }): CellData[] {
  const out: CellData[] = [];
  for (const a of accesses) {
    const phys = apply(sw, a.byteOffset);
    for (let b = 0; b < a.bytes; b += 4) {
      out.push({
        lane: a.laneId,
        bank: bankOfByte(phys + b),
        byteOffset: a.byteOffset + b,
        physOffset: phys + b,
        word: b >> 2,
      });
    }
  }
  return out;
}

export function ConflictMatrix() {
  const pat = activePattern.value;
  const accesses = currentAccesses.value;
  const sw = activeSwizzle.value;
  const elemBytes = activeTileDims.value.elemBytes;
  const way = maxConflict.value;
  const conflicts = currentConflicts.value;
  const lSel = laneSel.value;
  const bSel = bankSel.value;
  // Auto-drive cycle selection from the simulator when a consumer phase is
  // active (plan §D7). The user's manual `cycleSel` takes over outside a
  // consumer phase — clicking a cycle row still updates `cycleSel`, which
  // then drives rendering until the next consumer phase resumes.
  const cAtom = world.value.consumerAtom;
  const cy = cAtom ? cAtom.laneWave : cycleSel.value;
  const s = spec.value;

  const cells = useMemo(() => buildCells(accesses, sw), [accesses, sw]);
  const noSwizCells = useMemo(
    () => buildCells(accesses, effectiveSwizzle('none', elemBytes)),
    [accesses, elemBytes],
  );
  const arrival = useMemo(() => accessArrival(accesses, sw), [accesses, sw]);

  // Bank conflict depth per cell, and "winner" per bank in cycle 0.
  const bankLaneCount = useMemo(() => {
    const m = new Map<number, number>();
    for (const c of cells) m.set(c.bank, (m.get(c.bank) ?? 0) + 1);
    return m;
  }, [cells]);

  const W = PAD_L + 32 * CELL + 4;
  const H = PAD_T + HEAD_H + 32 * CELL + 4;

  const onCellClick = (lane: number, bank: number) => {
    laneSel.value = laneSel.value === lane ? null : lane;
    bankSel.value = bankSel.value === bank ? null : bank;
  };

  // Replay cycle strip geometry.
  const cyclesToDraw = Math.max(1, way);
  const stripRow = (cyc: number) => {
    const lanesInWave = [...arrival.wavePerLane.entries()]
      .filter(([, w]) => w === cyc)
      .map(([l]) => l)
      .sort((a, b) => a - b);
    return lanesInWave;
  };

  return (
    <div class="panel cm-panel">
      <h3>
        SIMT conflict matrix <small>— 32 lanes × 32 banks</small>
      </h3>
      <p class="cm__detail">
        pattern <code>{pat.id}</code> × <code>Swizzle&lt;{sw.B},{sw.M},{sw.S}&gt;</code> →{' '}
        <strong class={way > 1 ? 'cm__way--bad' : 'cm__way--ok'}>
          {way === 1 ? 'no conflict · 1 cycle' : `${way}-way · ${way} cycles`}
        </strong>
      </p>
      <div class="cm__svg-wrap">
        <svg class="cm__svg" width={W} height={H} viewBox={`0 0 ${W} ${H}`}>
          {/* bank column headers */}
          {Array.from({ length: 32 }, (_, b) => (
            <text
              key={`bh-${b}`}
              x={PAD_L + b * CELL + CELL / 2}
              y={PAD_T + HEAD_H - 4}
              text-anchor="middle"
              class={`cm__header ${bSel === b ? 'is-active' : ''}`}
            >
              {b}
            </text>
          ))}
          {/* lane row headers */}
          {Array.from({ length: 32 }, (_, l) => (
            <text
              key={`lh-${l}`}
              x={PAD_L - 4}
              y={PAD_T + HEAD_H + l * CELL + CELL - 4}
              text-anchor="end"
              class={`cm__header ${lSel === l ? 'is-active' : ''}`}
            >
              {l}
            </text>
          ))}
          {/* grid cells (empty) */}
          {Array.from({ length: 32 }, (_, l) =>
            Array.from({ length: 32 }, (_, b) => (
              <rect
                key={`g-${l}-${b}`}
                x={PAD_L + b * CELL}
                y={PAD_T + HEAD_H + l * CELL}
                width={CELL}
                height={CELL}
                class={`cm__empty ${lSel === l ? 'is-laneactive' : ''} ${bSel === b ? 'is-bankactive' : ''}`}
              />
            )),
          )}
          {/* accesses */}
          {cells.map((c, i) => {
            const depth = bankLaneCount.get(c.bank) ?? 1;
            const cyc = arrival.wavePerLane.get(c.lane) ?? 0;
            const isWinner = cyc === 0;
            const dim = cy !== cyc ? 0.25 : 1;
            return (
              <rect
                key={`c-${i}`}
                x={PAD_L + c.bank * CELL + 1}
                y={PAD_T + HEAD_H + c.lane * CELL + 1}
                width={CELL - 2}
                height={CELL - 2}
                class={`cm__cell cm__cell--w${Math.min(depth, 8)} ${isWinner ? 'is-winner' : ''}`}
                style={{ opacity: dim }}
                onClick={() => onCellClick(c.lane, c.bank)}
              >
                <title>
                  lane {c.lane} → bank {c.bank} · byte 0x{c.byteOffset.toString(16)}
                  {' → phys 0x'}{c.physOffset.toString(16)} · cycle {cyc}
                </title>
              </rect>
            );
          })}
          {/* NO-swizzle ghost overlay (only when shift held is not easy to
              wire in SVG — always show it dim for comparison) */}
          {s.swizzle !== 'none' &&
            noSwizCells.map((c, i) => (
              <rect
                key={`n-${i}`}
                x={PAD_L + c.bank * CELL + CELL / 2 - 1}
                y={PAD_T + HEAD_H + c.lane * CELL + CELL / 2 - 1}
                width={2}
                height={2}
                class="cm__ghost"
              >
                <title>
                  without swizzle: lane {c.lane} → bank {c.bank}
                </title>
              </rect>
            ))}
        </svg>
      </div>

      {/* Replay cycle strip */}
      <div class="cm__cycles">
        <div class="cm__cycles-head">
          replay cycles
          <span class="cm__cycles-hint">(click a cycle to isolate; [ ] to step)</span>
        </div>
        <div class="cm__cycles-row">
          {Array.from({ length: cyclesToDraw }, (_, cyc) => {
            const lanes = stripRow(cyc);
            return (
              <button
                key={`cs-${cyc}`}
                class={`cm__cycle ${cy === cyc ? 'is-active' : ''}`}
                onClick={() => (cycleSel.value = cyc)}
                title={`cycle ${cyc}: lanes [${lanes.slice(0, 8).join(', ')}${lanes.length > 8 ? '…' : ''}]`}
              >
                <span class="cm__cycle-lbl">c{cyc}</span>
                <span class="cm__cycle-bar">
                  {Array.from({ length: 32 }, (_, l) => (
                    <span
                      key={`cb-${cyc}-${l}`}
                      class={`cm__cycle-dot ${lanes.includes(l) ? 'is-on' : ''} ${lSel === l ? 'is-lanefocus' : ''}`}
                    />
                  ))}
                </span>
                <span class="cm__cycle-count">{lanes.length}</span>
              </button>
            );
          })}
        </div>
      </div>

      {conflicts.length > 0 && (
        <p class="cm__hot">
          hot banks: {conflicts.slice(0, 4).map((c) => (
            <code>{c.bank}×{c.way} </code>
          ))}
        </p>
      )}

      <TruthFooter
        verified
        models="32-lane × 32-bank grid for the active pattern + swizzle; replay-cycle row count matches analyzeConflicts().way; cycle auto-driver follows world.consumerAtom.laneWave when a consumer phase is active."
        schematic="multi-warp interference; bank port topology; sub-warp scheduling inside a collision."
        cite="patterns.ts · analyzeConflicts; accessArrival"
      />
    </div>
  );
}
