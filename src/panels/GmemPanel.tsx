// GMEM → SMEM (and acc → GMEM) in three tracks: the problem tensor with CTA
// tiles highlighted, an arrow column showing how each logical SMEM line is
// remapped under Swizzle<B,M,S>, and a physical SMEM line stack colored by
// source row. The arrows animate during `tma.load` / `cp.async` phases and
// reverse during `epilogue.tma.store` to tell the store story.
//
// This is the panel the user repeatedly asked for: "swizzle pattern line by
// line of global memory, detailed load / store pattern". It is deliberately
// placed above SmemPanel so the reader's eye travels GMEM → SMEM before
// diving into the SMEM-internal swizzle story.

import { useMemo } from 'preact/hooks';
import {
  activeSwizzleA,
  activeSwizzleB,
  blkM,
  blkMMult,
  blkN,
  blkNMult,
  blkK,
  clusterShape,
  ctaGrid,
  currentConsumerPhase,
  currentEpiloguePhase,
  currentProducerPhase,
  epiloguePhaseProgress,
  gmemA,
  gmemB,
  gmemC,
  inst,
  problemMMult,
  problemNMult,
  problemKMult,
  producerPhaseProgress,
  spec,
  world,
} from '../state';
import { apply } from '../swizzle';
import { bankOfByte } from '../swizzle';
import { bytesOf } from '../smem_budget';
import { TruthFooter } from './TruthFooter';

const PANEL_W = 920;
const OP_TRACK_H = 140;
const C_TRACK_H = 100;
const LEGEND_H = 92;
const GAP_Y = 10;

// Deterministic HSL hue per row index.
function hueFor(row: number): number {
  return (row * 47) % 360;
}

export function GmemPanel() {
  const i = inst.value;
  const s = spec.value;
  const gA = gmemA.value;
  const gB = gmemB.value;
  const gC = gmemC.value;
  const swA = activeSwizzleA.value;
  const swB = activeSwizzleB.value;
  const grid = ctaGrid.value;
  const [clusterM] = clusterShape.value;
  const pmm = problemMMult.value;
  const pnm = problemNMult.value;
  const pkm = problemKMult.value;
  const bmv = blkM.value;
  const bnv = blkN.value;
  const bkv = blkK.value;
  const producerPhase = currentProducerPhase.value;
  const epiloguePhase = currentEpiloguePhase.value;
  const consumerPhase = currentConsumerPhase.value;
  const pProg = producerPhaseProgress.value;
  const eProg = epiloguePhaseProgress.value;
  const w = world.value;
  const producerTransfer = w.producerTransfer;

  const totalH = OP_TRACK_H * 2 + C_TRACK_H + LEGEND_H + GAP_Y * 4;

  // Atom-level row stride — the span over which the swizzle permutes the
  // row-index bits. Using the CTA-tile stride instead would make every
  // row-pair land on a fresh 128 B line and the swizzle would appear as an
  // identity (all the yyy bits are below bit-7 in the canonical atoms). At
  // atom granularity multiple rows share each 128 B line so the word-to-bank
  // rotation is visible row-to-row — that is the story "swizzle pattern line
  // by line" is about.
  const elemBytesA = bytesOf(i.aDtypes[0]);
  const elemBytesB = bytesOf(i.bDtypes[0]);
  const rowStrideA = Math.max(
    4,
    Math.ceil((s.majorA === 'K' ? i.K : i.M) * elemBytesA),
  );
  const rowStrideB = Math.max(
    4,
    Math.ceil((s.majorB === 'K' ? i.K : i.N) * elemBytesB),
  );

  // Currently-loading K slab index. Producer drives this whenever the
  // simulator reports an in-flight producer transfer (TMA, cp.async, or any
  // sub-phase). When producer is idle, fall back to 0 — Phase 4 will wire
  // the consumer cursor here via world.consumerAtom.kSlab.
  const activeK = producerTransfer?.kSlab ?? 0;

  // Bank-legend active row: during a TMA load the linesLoaded counter marches
  // line-by-line 0 → linesTotal, so the legend animates row-by-row. Between
  // loads we freeze at the slab's base row (kSlab × atomsPerStage_K) so the
  // static picture still reads as "this slab". When the simulator has no
  // producer activity yet (tick 0 before any effect), we fall to 0.
  const linesPerSlabA = Math.max(1, w.producerTransfer?.linesTotal ?? 1);
  void linesPerSlabA; // reserved for future per-slab scaling (Phase 4)
  const legendActiveRow = producerTransfer
    ? producerTransfer.linesLoaded
    : activeK * Math.max(1, Math.floor(bmv / i.M));

  // Family-specific framing for the panel header + swizzle relevance.
  // wmma bypasses SMEM entirely (wmma.load_matrix_sync reads GMEM directly
  // into the warp-wide fragment), so the swizzle story is N/A. sm_80 mma
  // uses plain cp.async rather than TMA. Only sm_90/sm_100 use the full
  // TMA-into-swizzled-SMEM path that the panel's main story describes.
  const isWmma = i.family === 'wmma';
  const loadKind =
    i.family === 'wgmma' || i.family === 'tcgen05' || i.family === 'tcgen05.block_scaled'
      ? 'tma.load'
      : i.family === 'mma'
        ? 'cp.async'
        : 'wmma.load_matrix_sync';
  const pathLabel = isWmma ? 'GMEM ↔ .reg' : 'GMEM ↔ SMEM';

  return (
    <div class="panel gmem">
      <h3>
        {pathLabel} ({producerPhase ? `loading · ${loadKind}` : epiloguePhase ? 'storing' : 'idle'})
        <small>
          — problem <code>M={i.M * (bmv / i.M) * pmm}</code>×
          <code>N={i.N * (bnv / i.N) * pnm}</code>×
          <code>K={bkv * pkm}</code>
          {isWmma
            ? ' · warp-wide load_matrix_sync, no SMEM staging → swizzle N/A'
            : <> · each GMEM row → one SMEM line under <code>Swizzle&lt;{swA.B},{swA.M},{swA.S}&gt;</code></>}
        </small>
      </h3>
      <svg class="gmem__svg" width="100%" viewBox={`0 0 ${PANEL_W} ${totalH}`}>
        <defs>
          <marker id="gmem-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#7ca8d8" />
          </marker>
        </defs>

        {/* ----- Row A: operand A (M × K) ----- */}
        <OperandRow
          y={0}
          operand="A"
          dtype={gA.dtypeLabel}
          rows={gA.rows}
          cols={gA.cols}
          majorLabel={s.majorA}
          ctaRowsM={pmm}
          ctaColsK={pkm}
          ctaMySlot={{ m: 0, n: 0 }}
          ctaPeer={clusterM > 1 ? { m: 1, n: 0 } : null}
          swizzleLabel={`Swizzle<${swA.B},${swA.M},${swA.S}>`}
          swizzle={swA}
          activeK={activeK}
          blkOuter={bmv}
          blkK={bkv}
          kSlabs={pkm}
          rowStrideBytes={rowStrideA}
          phase={producerPhase ?? epiloguePhase ?? null}
          progress={producerPhase ? pProg : epiloguePhase ? eProg : 0}
          direction={producerPhase ? 'load' : epiloguePhase ? 'store' : 'idle'}
        />

        {/* ----- Row B: operand B (K × N) ----- */}
        <OperandRow
          y={OP_TRACK_H + GAP_Y}
          operand="B"
          dtype={gB.dtypeLabel}
          rows={gB.rows}
          cols={gB.cols}
          majorLabel={s.majorB}
          ctaRowsM={pnm}
          ctaColsK={pkm}
          ctaMySlot={{ m: 0, n: 0 }}
          ctaPeer={null}
          swizzleLabel={`Swizzle<${swB.B},${swB.M},${swB.S}>`}
          swizzle={swB}
          activeK={activeK}
          blkOuter={bnv}
          blkK={bkv}
          kSlabs={pkm}
          rowStrideBytes={rowStrideB}
          phase={producerPhase ?? epiloguePhase ?? null}
          progress={producerPhase ? pProg : epiloguePhase ? eProg : 0}
          direction={producerPhase ? 'load' : epiloguePhase ? 'store' : 'idle'}
        />

        {/* ----- Row C: epilogue destination ----- */}
        <CTrack
          y={(OP_TRACK_H + GAP_Y) * 2}
          h={C_TRACK_H}
          rows={gC.rows}
          cols={gC.cols}
          ctaRowsM={pmm}
          ctaColsN={pnm}
          hasTmem={i.accIn === 'tmem'}
          epiloguePhase={epiloguePhase}
          consumerPhase={consumerPhase}
          eProg={eProg}
          gridRowsM={grid.rowsM}
          gridColsN={grid.colsN}
          blkMMult={blkMMult.value}
          blkNMult={blkNMult.value}
          epilogueDrained={w.cTile?.epilogueDrained ?? null}
          epilogueStaged={w.cTile?.epilogueStaged ?? null}
        />

        {/* ----- Legend: one GMEM row byte-by-byte into 32 banks ----- */}
        <BankLegend
          y={(OP_TRACK_H + GAP_Y) * 2 + C_TRACK_H + GAP_Y}
          h={LEGEND_H}
          rowStrideBytes={rowStrideA}
          swizzle={swA}
          activeRow={legendActiveRow}
          elemBytes={gA.elemBytes}
        />
      </svg>

      <TruthFooter
        models="GMEM ↔ SMEM tile grid with this CTA highlighted; active slab driven by world.producerTransfer.kSlab (producer-side, leads the consumer by kStages-1); element-level bank legend animates line-by-line during active TMA phase; per-atom C-tile destination drains independently via world.cTile.epilogueDrained."
        schematic="TMA descriptor box-dim walking is atomic per phase (not multi-line walk); each GMEM row = one SMEM line under the chosen Swizzle; real kernels may use tiled TMA descriptors with non-trivial stride."
        cite="simulation.ts · world.producerTransfer; swizzle.ts · apply; PTX §9.7.9 cp.async.bulk.tensor"
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// OperandRow: GMEM tile (left) · arrows (middle) · SMEM physical lines (right)
// ---------------------------------------------------------------------------

function OperandRow(props: {
  y: number;
  operand: 'A' | 'B';
  dtype: string;
  rows: number;
  cols: number;
  majorLabel: string;
  ctaRowsM: number;
  ctaColsK: number;
  ctaMySlot: { m: number; n: number };
  ctaPeer: { m: number; n: number } | null;
  swizzleLabel: string;
  swizzle: { B: number; M: number; S: number };
  activeK: number;
  blkOuter: number;
  blkK: number;
  kSlabs: number;
  rowStrideBytes: number;
  phase: { kind: string } | null;
  progress: number;
  direction: 'load' | 'store' | 'idle';
}) {
  const {
    y, operand, dtype, majorLabel,
    ctaRowsM, ctaColsK, ctaMySlot, ctaPeer,
    swizzleLabel, swizzle,
    activeK, blkOuter, blkK: _blkK, kSlabs,
    rowStrideBytes,
    phase: _phase, progress, direction,
  } = props;

  // Split the panel into 2 horizontal tracks (GMEM tile | per-line bank grid).
  const leftW = 300;
  const rightW = 480;
  const pad = 24;
  const trackH = OP_TRACK_H;

  // ----- Left: GMEM tile (ctaRowsM × ctaColsK grid of CTA tiles) -----
  const cellW = (leftW - pad * 1.5) / Math.max(1, ctaColsK);
  const cellH = (trackH - 40) / Math.max(1, ctaRowsM);
  const leftX = pad;
  const leftYAxis = y + 24;

  // ----- Right: per-line bank map (the "line-by-line swizzle pattern") -----
  // Show up to DISPLAY_LINES logical lines from the active CTA tile slab.
  // For each line: draw 32 cells (one per 4-byte word), colored by the
  // destination SMEM bank under apply(Swizzle, byte).
  // This is the user-visible "swizzle pattern line by line of global memory":
  // adjacent GMEM rows take different bank patterns so ldmatrix warps don't
  // collide.
  const WORDS_PER_LINE = 32;
  const DISPLAY_LINES = Math.min(8, Math.max(1, Math.floor((trackH - 40) / 12)));
  // How many full SMEM lines one GMEM row covers. If rowStride < 128 multiple
  // rows share a line; if > 128 a row spans multiple lines. We display
  // `DISPLAY_LINES` rows from the active slab, indexing into the CTA tile.
  const totalRowsInSlab = Math.max(1, blkOuter);
  const rowsShown = Math.min(DISPLAY_LINES, totalRowsInSlab);
  const cellH2 = Math.max(6, Math.floor((trackH - 50) / Math.max(1, rowsShown)));
  const cellW2 = (rightW - 40) / WORDS_PER_LINE;
  const rightX = leftX + leftW + pad;

  // Build per-row bank assignments for the first `rowsShown` rows.
  interface LineEntry {
    row: number;
    banks: number[]; // one per word (0..31)
    hue: number;
  }
  const lines = useMemo<LineEntry[]>(() => {
    const out: LineEntry[] = [];
    for (let r = 0; r < rowsShown; r++) {
      const banks: number[] = [];
      for (let w = 0; w < WORDS_PER_LINE; w++) {
        const logical = r * rowStrideBytes + w * 4;
        const phys = apply(swizzle, logical);
        banks.push(bankOfByte(phys));
      }
      out.push({ row: r, banks, hue: hueFor(r) });
    }
    return out;
  }, [rowsShown, rowStrideBytes, swizzle.B, swizzle.M, swizzle.S]);

  const bankColor = (b: number) => `hsl(${(b * 360) / 32}, 65%, 55%)`;

  // Sweep: during load, rows fade in top-to-bottom; during store, they fade
  // OUT bottom-to-top (SMEM draining). Idle → static fully visible.
  const sweepCutoff = direction === 'idle' ? rowsShown : Math.max(0, Math.min(rowsShown, progress * rowsShown));

  return (
    <g class={`gmem__row gmem__row--${operand}`}>
      {/* label column */}
      <text x={8} y={y + 14} class="gmem__lbl">Operand {operand}</text>
      <text x={8} y={y + 28} class="gmem__lbl gmem__lbl--dim">
        {dtype} · {majorLabel}-major · {swizzleLabel}
      </text>

      {/* ----- Left: GMEM tile ----- */}
      <text x={leftX} y={y + 14} class="gmem__trackhead">GMEM ({operand === 'A' ? 'M×K' : 'K×N'})</text>
      <g transform={`translate(${leftX}, ${leftYAxis})`}>
        {Array.from({ length: ctaRowsM }, (_, tm) =>
          Array.from({ length: ctaColsK }, (_, tk) => {
            const isMe = tm === ctaMySlot.m && tk === ctaMySlot.n;
            const isPeer = ctaPeer && tm === ctaPeer.m && tk === ctaPeer.n;
            const isActiveSlab = isMe && tk === (activeK % kSlabs);
            return (
              <rect
                key={`g-${operand}-${tm}-${tk}`}
                x={tk * cellW}
                y={tm * cellH}
                width={cellW - 2}
                height={cellH - 2}
                class={`gmem__gcell ${isMe ? 'is-me' : ''} ${isPeer ? 'is-peer' : ''} ${isActiveSlab ? 'is-active' : ''}`}
              >
                <title>CTA ({tm},{tk}){isMe ? ' (this CTA)' : isPeer ? ' (cluster peer)' : ''}</title>
              </rect>
            );
          }),
        )}
        {/* Rainbow stripes on this CTA's tile — row hue carries through to
            the per-line bank map on the right. */}
        {(() => {
          const tm = ctaMySlot.m;
          const tk = ctaMySlot.n;
          const stripeCount = rowsShown;
          const stripeH = Math.max(1, (cellH - 4) / stripeCount);
          return Array.from({ length: stripeCount }, (_, ll) => (
            <rect
              key={`stripe-${operand}-${ll}`}
              x={tk * cellW + 2}
              y={tm * cellH + 2 + ll * stripeH}
              width={Math.max(0, cellW - 6)}
              height={Math.max(0, stripeH - 0.5)}
              fill={`hsl(${hueFor(ll)}, 70%, 55%)`}
              opacity={0.55}
            />
          ));
        })()}
      </g>
      <text x={leftX + leftW / 2 - pad / 2} y={y + trackH - 4} text-anchor="middle" class="gmem__axislbl gmem__axislbl--dim">
        CTA grid: {ctaRowsM} × {ctaColsK}{direction === 'load' ? ` · loading slab k=${activeK % kSlabs}` : direction === 'store' ? ' · storing C' : ''}
      </text>

      {/* ----- Right: per-line bank grid ----- */}
      <text x={rightX} y={y + 14} class="gmem__trackhead">
        SMEM lines (words → banks under swizzle)
      </text>
      <g transform={`translate(${rightX}, ${y + 24})`}>
        {lines.map((line, li) => {
          const visible = line.row < sweepCutoff;
          const op = direction === 'idle' ? 1 : visible ? 1 : 0.12;
          const rowY = li * cellH2;
          return (
            <g key={`line-${operand}-${li}`} opacity={op}>
              {/* Row label (hue dot + row number) on the left of the bank strip. */}
              <circle cx={-16} cy={rowY + cellH2 / 2} r={4} fill={`hsl(${line.hue}, 70%, 55%)`} />
              <text x={-8} y={rowY + cellH2 / 2 + 3} text-anchor="end" class="gmem__linelbl gmem__linelbl--dim">
                r{line.row}
              </text>
              {/* 32 word-cells, colored by destination bank. */}
              {line.banks.map((bank, w) => (
                <rect
                  key={`w-${operand}-${li}-${w}`}
                  x={w * cellW2}
                  y={rowY}
                  width={Math.max(1, cellW2 - 0.5)}
                  height={Math.max(1, cellH2 - 1)}
                  fill={bankColor(bank)}
                  stroke="#0b0e14"
                  stroke-width={0.3}
                >
                  <title>row {line.row} · word {w} · bank {bank}</title>
                </rect>
              ))}
            </g>
          );
        })}
        {/* Bottom axis: bank tick marks every 8 banks. */}
        <g transform={`translate(0, ${rowsShown * cellH2 + 2})`}>
          {Array.from({ length: 5 }, (_, k) => (
            <text
              key={`btick-${operand}-${k}`}
              x={k * 8 * cellW2}
              y={8}
              class="gmem__legendlbl"
            >
              {k * 8}
            </text>
          ))}
          <text x={WORDS_PER_LINE * cellW2} y={8} text-anchor="end" class="gmem__legendlbl">word→bank</text>
        </g>
      </g>
    </g>
  );
}

// ---------------------------------------------------------------------------
// CTrack: C-tile destination (epilogue store preview)
// ---------------------------------------------------------------------------

function CTrack(props: {
  y: number;
  h: number;
  rows: number;
  cols: number;
  ctaRowsM: number;
  ctaColsN: number;
  hasTmem: boolean;
  epiloguePhase: { kind: string } | null;
  consumerPhase: { kind: string } | null;
  eProg: number;
  gridRowsM: number;
  gridColsN: number;
  blkMMult: number;
  blkNMult: number;
  epilogueDrained: number[][] | null;
  epilogueStaged: number[][] | null;
}) {
  const {
    y, h, ctaRowsM, ctaColsN, hasTmem, epiloguePhase, gridRowsM, gridColsN,
    blkMMult, blkNMult, epilogueDrained, epilogueStaged,
  } = props;
  const pad = 24;

  const cellW = (PANEL_W - pad * 2) / Math.max(1, ctaColsN);
  const cellH = (h - 40) / Math.max(1, ctaRowsM);

  const activeStore = epiloguePhase !== null;
  const phaseKind = epiloguePhase?.kind ?? '';
  // Staging (stg_smem) fills the intermediate SMEM-C region; drain
  // (tma.store) empties it to GMEM. We render each as a per-atom sub-cell
  // tint so users see the row-major sweep in both phases.
  void phaseKind;

  // The CTA's C tile is subdivided into blkMMult × blkNMult atom sub-cells
  // — matching Row 3 of TileHierarchyPanel. Each sub-cell's green drain
  // fill reflects world.cTile.epilogueDrained[am][an] and shows the
  // row-major 0→1 sweep during tma.store. During stg_smem we use a
  // slightly different tint (yellow) for epilogueStaged.
  const bmLocal = Math.max(1, blkMMult);
  const bnLocal = Math.max(1, blkNMult);

  return (
    <g class="gmem__ctrack">
      <text x={8} y={y + 14} class="gmem__lbl">C tile → GMEM</text>
      <text x={8} y={y + 28} class="gmem__lbl gmem__lbl--dim">
        {hasTmem ? 'tcgen05.ld → .reg → SMEM staging → tma.store → GMEM' : '.reg → SMEM staging → tma.store → GMEM (or direct stg for sm_80)'}
      </text>

      <g transform={`translate(${pad}, ${y + 32})`}>
        {Array.from({ length: ctaRowsM }, (_, tm) =>
          Array.from({ length: ctaColsN }, (_, tn) => {
            const isMe = tm === 0 && tn === 0;
            return (
              <rect
                key={`c-${tm}-${tn}`}
                x={tn * cellW}
                y={tm * cellH}
                width={cellW - 2}
                height={cellH - 2}
                class={`gmem__ccell ${isMe ? 'is-me' : ''} ${isMe && activeStore ? 'is-storing' : ''}`}
              >
                <title>C tile ({tm},{tn}){isMe ? ' (this CTA)' : ''}</title>
              </rect>
            );
          }),
        )}
        {/* Our CTA's C tile — subdivided into blkMMult × blkNMult atom
            sub-cells, each with a per-atom drain overlay. */}
        {(() => {
          const tm = 0;
          const tn = 0;
          const subW = (cellW - 2) / bnLocal;
          const subH = (cellH - 2) / bmLocal;
          return Array.from({ length: bmLocal }, (_, am) =>
            Array.from({ length: bnLocal }, (_, an) => {
              const drained = epilogueDrained?.[am]?.[an] ?? 0;
              const staged = epilogueStaged?.[am]?.[an] ?? 0;
              const x0 = tn * cellW + an * subW;
              const y0 = tm * cellH + am * subH;
              return (
                <g key={`c-sub-${am}-${an}`}>
                  {/* Staged SMEM-C fill (yellow, faint) — visible during
                      stg_smem and persists after. */}
                  {staged > 0 && (
                    <rect
                      x={x0}
                      y={y0}
                      width={subW - 1}
                      height={subH - 1}
                      fill="#e0cf7a"
                      opacity={0.35 * staged}
                    >
                      <title>
                        atom ({am},{an}) SMEM-C staged {Math.round(staged * 100)}%
                      </title>
                    </rect>
                  )}
                  {/* Drained to GMEM (green) — overrides staged where active. */}
                  {drained > 0 && (
                    <rect
                      x={x0}
                      y={y0}
                      width={subW - 1}
                      height={subH - 1}
                      fill="#7ec699"
                      opacity={0.7 * drained}
                    >
                      <title>
                        atom ({am},{an}) drained to GMEM {Math.round(drained * 100)}%
                      </title>
                    </rect>
                  )}
                  {/* Sub-cell outline so the atom grid is visible even when empty. */}
                  <rect
                    x={x0}
                    y={y0}
                    width={subW - 1}
                    height={subH - 1}
                    fill="transparent"
                    stroke="#2a2f3a"
                    stroke-width={0.5}
                  />
                </g>
              );
            }),
          );
        })()}
      </g>
      <text x={PANEL_W - pad} y={y + h - 4} text-anchor="end" class="gmem__axislbl gmem__axislbl--dim">
        C problem grid: {gridRowsM} × {gridColsN} · atom grid: {bmLocal} × {bnLocal}
      </text>
    </g>
  );
}

// ---------------------------------------------------------------------------
// BankLegend: one GMEM row, 32 words, colored by SMEM bank under swizzle
// ---------------------------------------------------------------------------

function BankLegend(props: {
  y: number;
  h: number;
  rowStrideBytes: number;
  swizzle: { B: number; M: number; S: number };
  activeRow: number;
  elemBytes: number;
}) {
  const { y, h, rowStrideBytes, swizzle, activeRow } = props;
  const pad = 24;
  const trackW = PANEL_W - pad * 2;

  // Words per legend line: at most 32 (one full line), or fewer if the row
  // stride is shorter (small tileK fp16 kernels have rowStride=32).
  const wordsInRow = Math.max(1, Math.min(32, Math.floor(rowStrideBytes / 4)));
  const cellW = trackW / wordsInRow;
  const cellH = Math.max(12, Math.min(28, h - 40));

  // For each word in the row: compute its logical byte offset and the swizzle
  // remap; color by target bank id (0..31).
  const items = useMemo(() => {
    return Array.from({ length: wordsInRow }, (_, w) => {
      const logical = activeRow * rowStrideBytes + w * 4;
      const phys = apply(swizzle, logical);
      const bank = bankOfByte(phys);
      return { w, bank, logical, phys };
    });
  }, [wordsInRow, swizzle.B, swizzle.M, swizzle.S, activeRow, rowStrideBytes]);

  // Bank-color palette: 32 distinct hues.
  const bankColor = (b: number) => `hsl(${(b * 360) / 32}, 65%, 55%)`;

  return (
    <g class="gmem__legend">
      <text x={8} y={y + 14} class="gmem__lbl">Element-level load pattern</text>
      <text x={8} y={y + 28} class="gmem__lbl gmem__lbl--dim">
        one GMEM row ({wordsInRow} × 4B words) → 32 SMEM banks · hue = target bank
      </text>

      <g transform={`translate(${pad}, ${y + 34})`}>
        {items.map((it) => (
          <g key={`leg-${it.w}`}>
            <rect
              x={it.w * cellW}
              y={0}
              width={cellW - 1}
              height={cellH}
              fill={bankColor(it.bank)}
              stroke="#0b0e14"
              stroke-width={0.3}
            >
              <title>word {it.w} · byte {it.logical} → phys byte {it.phys} · bank {it.bank}</title>
            </rect>
            <text
              x={it.w * cellW + cellW / 2}
              y={cellH + 12}
              text-anchor="middle"
              class="gmem__legendlbl"
            >
              b{it.bank}
            </text>
          </g>
        ))}
      </g>
    </g>
  );
}
