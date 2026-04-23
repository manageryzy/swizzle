// ---------------------------------------------------------------------------
// v4 simulator — single source of truth for every panel's "what's happening
// at tick t" question.
//
// This is the Phase 1 SKELETON. It exposes `simulate(input: SimInput):
// SimResult` with:
//   - `streams`     populated (producer / consumer / epilogue phase lists).
//   - `summary`     fully populated (slabCount, atomsPerStage_K, familyShape,
//                   variant, extras, hasRing, hasMbar, …).
//   - `worldAt()`   returns a WorldState whose `active`, `progress`, `ring`
//                   and `mbar` fields are all populated; the richer fields
//                   (producerTransfer, consumerAtom, cTile, warps, cluster,
//                   auxiliary) are stubbed as null/empty for later phases.
//
// The effect-driven builder and the `worldAt(tick)` infrastructure are both
// wired end-to-end here so later phases only add new effect kinds rather
// than re-plumbing the query path.
//
// Phase 1 intentionally emits the same per-stream phase COUNT that v3's
// `emitTimeline()` does: producer.length === consumer.length. The correct
// per-family asymmetry (slabCount producer vs slabCount × atomsPerStage_K
// consumer) is Phase 2. See TODO(phase2) below.
//
// Design references:
//   plan §A (three nested K-walk levels)
//   plan §B (WorldState shape)
//   plan §C (SimResult / SimSummary / Effect / interpolation)
//   plan §K (per-family execution semantics)
//   plan §L (warp role table)
//   plan §N (invariants 1..5 for the skeleton)
//   plan §R (11-step bootstrapping order)
// ---------------------------------------------------------------------------

import type { Dtype, InstSpec, Major, OperandSource, PipelineMode } from './instructions';
import type { SwizzleKind } from './swizzle';
import { effectiveSwizzle } from './swizzle';
import { bytesOf } from './smem_budget';
import { type TimelinePhase } from './pipeline_state';
import { LDMATRIX_X4_N, analyzeConflicts } from './patterns';
import { tileDimsFor } from './tile_dims';

// ---------------------------------------------------------------------------
// Part B — Types
// ---------------------------------------------------------------------------

export type FamilyShape = 'tma-warpspec' | 'cpasync-mma' | 'wmma-direct';
export type Variant = 'ss' | 'rs' | 'ts';

export interface SimInput {
  inst: InstSpec;
  majorA: Major;
  majorB: Major;
  swizzle: SwizzleKind;
  dtypeA: Dtype;
  dtypeB: Dtype;
  accDtype: Dtype;
  aSource: OperandSource;
  blkMMult: number;
  blkNMult: number;
  /** BLK_K — the CTA K-tile in elements. */
  tileK: number;
  /** Pipeline ring depth. Ignored for coupled family. */
  kStages: number;
  problemMMult: number;
  problemNMult: number;
  problemKMult: number;
}

export interface SimSummary {
  slabCount: number;               // = problemK / BLK_K
  atomsPerStage_K: number;         // = BLK_K / atomK
  atomsPerStage_MN: number;        // = blkMMult × blkNMult
  consumerItersTotal: number;      // = slabCount × atomsPerStage_K
  linesPerSlab_A: number;          // = ceil(BLK_M × BLK_K × elemA / 128)
  linesPerSlab_B: number;          // = ceil(BLK_N × BLK_K × elemB / 128)
  /** Combined slab transfer size (A + B), 128 B lines. Used by TMA/cp.async. */
  linesPerSlab_AB: number;
  /** ldmatrix (RS) atom A transfer: ceil(atomM × atomK × elemA / 128). */
  linesPerAtom_ldmatrixA: number;
  /** tcgen05.cp slab transfer (SMEM-A → TMEM): ceil(atomM × atomK × elemA / 128). */
  linesPerSlab_tcgen05cp: number;
  /** Sparse 2:4 metadata TMA per slab (~BLK_M × BLK_K / 4 bytes). */
  linesPerSlab_metadata: number;
  /** Block-scaled SFA + SFB combined TMA per slab (BLK_{M,N} × BLK_K / scaleGroupSize). */
  linesPerSlab_scale: number;
  cCells: { m: number; n: number }; // = (blkMMult, blkNMult)
  hasRing: boolean;                // pipelineMode === 'warpspec'
  hasMbar: boolean;                // same
  familyShape: FamilyShape;
  variant: Variant;
  extras: {
    sparse: boolean;
    blockScaled: boolean;
    ctaGroup: 1 | 2;
    warpSpecialized: boolean;
  };
  pipelineMode: PipelineMode;
  /** Consumer-side bank conflict depth (max ways) under the active A-side
   *  swizzle and the canonical consumer access pattern (ldmatrix.x4.N). Drives
   *  `consumerAtom.laneWave` cycling and ConflictMatrix auto-cycle. Min 1. */
  maxWaysConsumer: number;
  /** The value `cTile.accumulated[m][n]` reaches once the mainloop finishes
   *  for every (m, n). Each consumer phase fires on ALL (m, n) atoms once, so
   *  the cap is `slabCount × atomsPerStage_K = consumerItersTotal` — not
   *  divided by atomsPerStage_MN. Panels use this to normalize fill opacity. */
  maxAccumulatedPerAtom: number;
}

export interface RingSlotState {
  stage: number;
  slice: number;                   // K-slab index currently held in this stage
  role: 'consume' | 'fill' | 'hold' | 'drained' | 'empty';
  fillFrac: number;                // 0..1 while 'fill' is in progress
}

export interface MbarState {
  stage: number;
  state: 'full' | 'empty';
  lastArriveTick: number;
  lastWaitTick: number;
}

// --- Populated in later phases — see phase N comments per field. ---------
export interface ProducerTransferState {
  kind: 'tma' | 'cpasync' | 'wmma-load' | 'ldmatrixA' | 'tcgen05-cp' | 'metadata' | 'scale';
  kSlab: number;
  operand: 'A' | 'B' | 'AB' | 'meta' | 'scaleA' | 'scaleB';
  stage: number;
  linesLoaded: number;
  linesTotal: number;
  bytesInFlight: number;
}

export interface ConsumerAtomState {
  kSlab: number;
  kStep: number;
  kAtomInSlab: number;
  stage: number;
  atomM: number;
  atomN: number;
  atomFlatIdx: number;
  laneWave: number;
  maxWays: number;
}

export interface CTileState {
  accumulated: number[][];
  epilogueStaged: number[][];
  epilogueDrained: number[][];
}

export interface WarpState {
  warpIdx: number;
  role: 'producer' | 'consumer' | 'epilogue' | 'idle';
  fragment: {
    kind: 'A' | 'B' | 'C' | 'none';
    atomM?: number;
    atomN?: number;
    kAtomInSlab?: number;
  };
}

export interface ClusterState {
  thisCtaRole: 'leader' | 'peer';
  peerActive: boolean;
  sharedLoad: boolean;
}

export interface AuxiliaryState {
  metadata: boolean;
  scaleA: boolean;
  scaleB: boolean;
}

export interface WorldState {
  tick: number;

  // ---------- Stream currency ----------
  active: {
    producer: TimelinePhase | null;
    consumer: TimelinePhase | null;
    epilogue: TimelinePhase | null;
  };
  progress: {
    producer: number;
    consumer: number;
    epilogue: number;
  };

  // ---------- Ring state (warpspec: length kStages; coupled mma: 1; wmma: 0) ----------
  ring: RingSlotState[];

  // ---------- Mbarrier events (warpspec only; coupled families: length 0). ----------
  mbar: MbarState[];

  // ---------- Populated in phase 3 ----------
  producerTransfer: ProducerTransferState | null;

  // ---------- Populated in phase 4 ----------
  consumerAtom: ConsumerAtomState | null;

  // ---------- Populated in phase 5 ----------
  cTile: CTileState | null;

  // ---------- Populated in phase 6 (variants & warp-specialization) ----------
  warps: WarpState[];
  cluster: ClusterState | null;
  auxiliary: AuxiliaryState;
}

export interface Effect {
  atTick: number;
  apply: (state: WorldState) => void;
  interpolateUntil?: number;
  interpolate?: (state: WorldState, frac: number) => void;
}

export interface SimStreams {
  producer: TimelinePhase[];
  consumer: TimelinePhase[];
  epilogue: TimelinePhase[];
}

export interface SimResult {
  totalTicks: number;
  streams: SimStreams;
  effects: Effect[];
  worldAt: (tick: number) => WorldState;
  summary: SimSummary;
}

// ---------------------------------------------------------------------------
// Part R step 2 — deriveSummary
// ---------------------------------------------------------------------------

function pickVariant(aSource: OperandSource): Variant {
  if (aSource === 'rmem') return 'rs';
  if (aSource === 'tmem') return 'ts';
  return 'ss';
}

function pickFamilyShape(inst: InstSpec): FamilyShape {
  if (inst.family === 'wmma') return 'wmma-direct';
  if (inst.pipelineMode === 'warpspec') return 'tma-warpspec';
  return 'cpasync-mma';
}

export function deriveSummary(input: SimInput): SimSummary {
  const i = input.inst;
  const slabCount = Math.max(1, input.problemKMult);
  const atomsPerStage_K = Math.max(1, Math.ceil(input.tileK / i.K));
  const atomsPerStage_MN = Math.max(1, input.blkMMult * input.blkNMult);

  const BLK_M = i.M * input.blkMMult;
  const BLK_N = i.N * input.blkNMult;
  const BLK_K = input.tileK;
  const elemA = bytesOf(input.dtypeA);
  const elemB = bytesOf(input.dtypeB);
  // 128-byte line granularity (SMEM cache line / swizzle atom).
  const linesPerSlab_A = Math.max(1, Math.ceil((BLK_M * BLK_K * elemA) / 128));
  const linesPerSlab_B = Math.max(1, Math.ceil((BLK_N * BLK_K * elemB) / 128));
  // Combined TMA (A + B) lands in the slab ring stage under warpspec.
  const linesPerSlab_AB = linesPerSlab_A + linesPerSlab_B;
  // Sub-phase line counts (per plan §K data-transfer shapes).
  const atomM = i.M;
  const atomN = i.N;
  const atomK = i.K;
  // ldmatrix A (RS): atomM × atomK × bytesA per k-atom.
  const linesPerAtom_ldmatrixA = Math.max(1, Math.ceil((atomM * atomK * elemA) / 128));
  // tcgen05.cp (TS): atomM × atomK × bytesA per slab (moving SMEM-A → TMEM).
  const linesPerSlab_tcgen05cp = Math.max(1, Math.ceil((atomM * atomK * elemA) / 128));
  // Sparse 2:4 metadata: BLK_M × BLK_K / 4 bytes (2 bits per element, fp16 baseline).
  const linesPerSlab_metadata = Math.max(1, Math.ceil((BLK_M * BLK_K) / 4 / 128));
  // Block-scaled SFA + SFB: ~(BLK_M + BLK_N) × BLK_K / scaleGroupSize bytes; group size=32.
  const scaleGroup = 32;
  const linesPerSlab_scale = Math.max(
    1,
    Math.ceil(((BLK_M + BLK_N) * BLK_K) / scaleGroup / 128),
  );
  // Reference atomN to avoid unused-local (kept for future per-operand sizing).
  void atomN;

  const familyShape = pickFamilyShape(i);
  const variant = pickVariant(input.aSource);
  const hasRing = i.pipelineMode === 'warpspec';
  const hasMbar = hasRing;

  // Consumer-side max conflict ways under the active A-side swizzle. We use
  // the canonical ldmatrix.x4.N pattern (K-major A, fp16 default) as the
  // consumer access probe — it's the one the ConflictMatrix panel displays
  // for SS/RS/TS variants. The A-side tile dims determine rowStride/tileBytes
  // at `atomM × atomK` granularity (per plan feedback: swizzle acts
  // within-line, so atom-level row stride is the right choice).
  const atomDimsA = tileDimsFor(i, 'A', input.majorA, 1);
  const atomRowStrideBytes = Math.max(4, Math.ceil(i.K * elemA));
  const swizzleEff = effectiveSwizzle(input.swizzle, elemA);
  const consumerAccesses = LDMATRIX_X4_N.accesses({
    rowStrideBytes: atomRowStrideBytes,
    tileBytes: Math.max(atomDimsA.tileBytes, atomRowStrideBytes * 8),
  });
  const consumerConflicts = analyzeConflicts(consumerAccesses, swizzleEff);
  const maxWaysConsumer = consumerConflicts.length === 0
    ? 1
    : consumerConflicts[0].way;

  return {
    slabCount,
    atomsPerStage_K,
    atomsPerStage_MN,
    consumerItersTotal: slabCount * atomsPerStage_K,
    linesPerSlab_A,
    linesPerSlab_B,
    linesPerSlab_AB,
    linesPerAtom_ldmatrixA,
    linesPerSlab_tcgen05cp,
    linesPerSlab_metadata,
    linesPerSlab_scale,
    cCells: { m: input.blkMMult, n: input.blkNMult },
    hasRing,
    hasMbar,
    familyShape,
    variant,
    extras: {
      sparse: !!i.sparse,
      blockScaled: i.family === 'tcgen05.block_scaled',
      ctaGroup: (i.ctaGroup ?? 1) as 1 | 2,
      warpSpecialized: !!i.warpSpecialized,
    },
    pipelineMode: i.pipelineMode,
    maxWaysConsumer,
    maxAccumulatedPerAtom: slabCount * atomsPerStage_K,
  };
}

// ---------------------------------------------------------------------------
// Part M — Timing model (exported so tests can override).
// ---------------------------------------------------------------------------

export const TICKS = {
  TMA: 4,
  CPASYNC: 4,
  WMMA_LOAD: 2,
  LDMATRIX: 2,
  TCG_CP: 3,
  METADATA: 2,
  SCALE: 2,
  MMA: 2,           // per MN atom — consumer phase = MMA × atomsPerStage_MN.
  TCG_LD: 3,
  STG_SMEM: 2,      // baseline — scales with blkMMult×blkNMult via max(2, ceil(/4))
  TMA_STORE: 2,
  ST_GLOBAL: 2,
} as const;

/** Rendering cap per stream. Beyond this, trailing phases collapse. Plan §O5. */
export const MAX_PHASES_PER_STREAM = 8;

// ---------------------------------------------------------------------------
// Part R step 3 — emitPhases (family-aware, plan §K1..§K9)
// ---------------------------------------------------------------------------
//
// Each family emits distinct producer sub-phase structures per slab:
//   wmma:    1 wmma.load per slab                 (coupled; per-k-step pattern)
//   mma:     1 cp.async + atomsPerStage_K ldmatrix per slab (coupled)
//   wgmma:   1 tma.load per slab                  (warpspec SS)
//            + atomsPerStage_K ldmatrix if RS
//   tcgen05: 1 tma.load per slab                  (warpspec SS)
//            + 1 tcgen05.cp if TS
//            + 1 metadata if sparse
//            + 1 scale if block_scaled
// Consumer is ALWAYS slabCount × atomsPerStage_K mma steps.
//
// Each consumer phase has duration MMA_TICKS × atomsPerStage_MN so the MN
// sub-atom work (populated in Phase 4) has visible tick budget.

type ProducerKind =
  | 'tma.load'
  | 'cp.async'
  | 'ldmatrix'
  | 'wmma.load'
  | 'tcgen05.cp'
  | 'metadata'
  | 'scale';

interface EmitCtx {
  input: SimInput;
  summary: SimSummary;
  consumerKind: 'wgmma.step' | 'tcgen05.mma.step';
  mmaPhaseTicks: number;
  idPrefix: string;
}

function mmaStepKind(i: InstSpec): 'wgmma.step' | 'tcgen05.mma.step' {
  return i.family === 'tcgen05' || i.family === 'tcgen05.block_scaled'
    ? 'tcgen05.mma.step'
    : 'wgmma.step';
}

function accInLabel(i: InstSpec): string {
  return i.accIn === 'rmem' ? '.reg' : i.accIn.toUpperCase();
}

// Producer sub-descriptor — everything we need to turn one abstract sub-phase
// into a real TimelinePhase.
interface ProducerSub {
  kind: ProducerKind;
  label: (slab: number, atomInSlab?: number) => string;
  description: (slab: number, atomInSlab?: number) => string;
  ticks: number;
  producerSub: NonNullable<TimelinePhase['producerSub']>;
  /** If true, emit once per slab. If false, emit atomsPerStage_K times per slab. */
  perSlab: boolean;
  /** Which ring stage this sub-phase targets (perSlab only). Defaults to slab % kStages. */
  stageOverride?: (slab: number) => number;
}

function describeProducerSubs(ctx: EmitCtx): ProducerSub[] {
  const i = ctx.input.inst;
  const fam = i.family;
  const aSource = ctx.input.aSource;
  const subs: ProducerSub[] = [];

  // === Coupled families ===
  if (fam === 'wmma') {
    // §K1. wmma: one gmem→.reg load per slab; no SMEM, no ring.
    subs.push({
      kind: 'wmma.load',
      label: (s) => s === 0 ? 'wmma.load_matrix_sync A,B (prime)' : `wmma.load slab ${s}`,
      description: (s) =>
        `Warp-wide wmma.load_matrix_sync fetches A/B fragments for K slab ${s} directly from GMEM/SMEM into warp registers. No ring, no mbarrier.`,
      ticks: TICKS.WMMA_LOAD,
      producerSub: 'wmma-load',
      perSlab: true,
    });
    return subs;
  }

  if (fam === 'mma') {
    // §K2. sm_80 mma: cp.async → SMEM, then atomsPerStage_K ldmatrix per slab.
    subs.push({
      kind: 'cp.async',
      label: (s) => s === 0 ? 'cp.async → SMEM (A,B)' : `cp.async slab ${s}`,
      description: (s) =>
        `cp.async loads one K slab (A+B) for slice ${s} into SMEM. Same warp then issues ldmatrix per k-atom.`,
      ticks: TICKS.CPASYNC,
      producerSub: 'cpasync',
      perSlab: true,
      stageOverride: () => 0, // synthetic single-stage
    });
    subs.push({
      kind: 'ldmatrix',
      label: (s, a) => `ldmatrix slab ${s} · k-atom ${a}`,
      description: (s, a) =>
        `Warp loads atom-K slice ${a} of slab ${s} from swizzled SMEM into per-lane registers (SMEM → .reg).`,
      ticks: TICKS.LDMATRIX,
      producerSub: 'ldmatrixA',
      perSlab: false,
      stageOverride: () => 0,
    });
    return subs;
  }

  // === Warp-specialized families ===
  // §K3 SS / §K4 RS / §K5 TS — plus cg2 (multicast), sparse, block_scaled.

  // One tma.load per slab for A+B.
  subs.push({
    kind: 'tma.load',
    label: (s) => {
      const stage = s % ctx.input.kStages;
      return s === 0 ? `tma.load stage ${stage} (prime)` : `tma.load stage ${stage} · k-slab ${s}`;
    },
    description: (s) =>
      `Producer warpgroup issues cp.async.bulk.tensor (TMA) for K slab ${s} into ring stage ${s % ctx.input.kStages}. ` +
      `Arrives on mbar[${s % ctx.input.kStages}].full when complete.`,
    ticks: TICKS.TMA,
    producerSub: 'tma',
    perSlab: true,
  });

  // Sparse: one metadata tma per slab.
  if (i.sparse) {
    subs.push({
      kind: 'metadata',
      label: (s) => `tma.meta slab ${s}`,
      description: (s) =>
        `Sparse 2:4 metadata tensor TMA for slab ${s} — small companion load (1/4 of dense A bytes) used by the sparse mma.`,
      ticks: TICKS.METADATA,
      producerSub: 'metadata',
      perSlab: true,
    });
  }

  // Block-scaled: one scale tma per slab (SFA + SFB lumped here).
  if (fam === 'tcgen05.block_scaled') {
    subs.push({
      kind: 'scale',
      label: (s) => `tma.scale slab ${s}`,
      description: (s) =>
        `Block-scaled scale-factor tensors (SFA + SFB) TMA for slab ${s}. Size ~ BLK_{M,N} × (BLK_K / scaleGroupSize).`,
      ticks: TICKS.SCALE,
      producerSub: 'scale',
      perSlab: true,
    });
  }

  // TS variant: tcgen05.cp SMEM-A → TMEM-A, one per slab, AFTER tma.load.
  if (aSource === 'tmem') {
    subs.push({
      kind: 'tcgen05.cp',
      label: (s) => `tcgen05.cp slab ${s}`,
      description: (s) =>
        `Copy A slab ${s} from SMEM to TMEM (tcgen05.cp). A operand then consumed from TMEM via the TS descriptor.`,
      ticks: TICKS.TCG_CP,
      producerSub: 'tcgen05-cp',
      perSlab: true,
    });
  }

  // RS variant: atomsPerStage_K ldmatrix-A sub-phases per slab, hoisted
  // just-in-time before each consumer phase.
  if (aSource === 'rmem') {
    subs.push({
      kind: 'ldmatrix',
      label: (s, a) => `ldmatrix.A slab ${s} · k-atom ${a}`,
      description: (s, a) =>
        `RS variant: ldmatrix loads atom-K slice ${a} of slab ${s}'s A operand from SMEM into .reg, hoisted immediately before its consuming wgmma.step.`,
      ticks: TICKS.LDMATRIX,
      producerSub: 'ldmatrixA',
      perSlab: false,
    });
  }

  return subs;
}

// Append one phase at cursor `t`; advance cursor; return the phase.
function pushPhase(
  out: TimelinePhase[],
  phase: TimelinePhase,
): TimelinePhase {
  out.push(phase);
  return phase;
}

function consumerLabel(
  ctx: EmitCtx,
  kSlab: number,
  kAtomInSlab: number,
  kStepFlat: number,
  totalConsumer: number,
): string {
  const stage = kSlab % ctx.input.kStages;
  return `mma k=${kStepFlat + 1}/${totalConsumer} · slab ${kSlab}/atom ${kAtomInSlab} · stage ${stage}`;
}

function consumerDescription(
  ctx: EmitCtx,
  kSlab: number,
  kAtomInSlab: number,
): string {
  const stage = kSlab % ctx.input.kStages;
  return (
    `Tensor core consumes atom-K slice ${kAtomInSlab} of slab ${kSlab} from stage ${stage}; ` +
    `accumulator lands in ${accInLabel(ctx.input.inst)}. ` +
    `atomsPerStage_MN=${ctx.summary.atomsPerStage_MN} MN atoms fire within this phase.`
  );
}

function emitWarpspec(ctx: EmitCtx): { streams: SimStreams; totalTicks: number } {
  const { summary, input } = ctx;
  const { slabCount, atomsPerStage_K } = summary;
  const kStages = input.kStages;
  const consumerItersTotal = slabCount * atomsPerStage_K;
  const i = input.inst;
  const idPrefix = ctx.idPrefix;

  const producer: TimelinePhase[] = [];
  const consumer: TimelinePhase[] = [];
  const epilogue: TimelinePhase[] = [];

  const subs = describeProducerSubs(ctx);
  const perSlabSubs = subs.filter((s) => s.perSlab);
  const perAtomSubs = subs.filter((s) => !s.perSlab);

  // --- Display cap: only render first MAX_PHASES_PER_STREAM slabs of producer
  // and first MAX_PHASES_PER_STREAM consumer iters. Everything beyond collapses
  // into one trailing phase per stream that covers the omitted tick budget.
  const producerSlabsDisplayed = Math.min(slabCount, MAX_PHASES_PER_STREAM);
  const producerSlabsTail = slabCount - producerSlabsDisplayed;
  const consumerItersDisplayed = Math.min(consumerItersTotal, MAX_PHASES_PER_STREAM);
  const consumerItersTail = consumerItersTotal - consumerItersDisplayed;

  // Emit producer slabs. For each slab, emit each perSlab sub-phase back-to-back.
  // (Per-atom RS sub-phases are hoisted next to their consumer phases; handled
  // in the consumer loop below.)
  //
  // Back-pressure: slab s's tma.load cannot start until consumer[s-kStages, last]
  // has finished (ring-full stall). We seed producer eagerly then run a fixup
  // pass after consumer is built (same strategy as v3's emitTimeline).

  // Track per-slab producer end tick (the moment all per-slab sub-phases
  // complete — consumer preconditions gate on this, not on tma.load.end alone).
  const perSlabSubEndForSlab: number[] = new Array(producerSlabsDisplayed).fill(0);

  let pCursor = 0;
  for (let s = 0; s < producerSlabsDisplayed; s++) {
    const stage = s % kStages;
    let slabStart = pCursor;
    for (const sub of perSlabSubs) {
      const st = sub.stageOverride ? sub.stageOverride(s) : stage;
      const start = slabStart;
      const end = start + sub.ticks;
      pushPhase(producer, {
        id: `${idPrefix}.${sub.kind}.prod.s${s}`,
        kind: sub.kind,
        startTick: start,
        endTick: end,
        label: sub.label(s),
        description: sub.description(s),
        iter: s,
        stage: st,
        kSlab: s,
        producerSub: sub.producerSub,
      });
      slabStart = end;
    }
    perSlabSubEndForSlab[s] = slabStart;
    pCursor = slabStart;
  }

  // Tail collapse for producer.
  if (producerSlabsTail > 0) {
    // Tail covers all remaining slabs' per-slab subs. Duration scaled.
    const subBudget = perSlabSubs.reduce((a, b) => a + b.ticks, 0);
    const tailTicks = Math.max(1, subBudget); // at least render a marker bar
    const start = pCursor;
    const end = start + tailTicks;
    pushPhase(producer, {
      id: `${idPrefix}.prod.tail`,
      kind: perSlabSubs[0]?.kind ?? 'tma.load',
      startTick: start,
      endTick: end,
      label: `× ${producerSlabsTail} more slabs`,
      description: `${producerSlabsTail} additional producer slabs collapsed for legibility. Each fills stage (slab mod ${kStages}) on the ring.`,
      iter: producerSlabsDisplayed,
      stage: producerSlabsDisplayed % kStages,
      kSlab: producerSlabsDisplayed,
      collapsedCount: producerSlabsTail,
      producerSub: perSlabSubs[0]?.producerSub ?? 'tma',
    });
    pCursor = end;
  }

  // --- Consumer stream: slabCount × atomsPerStage_K phases, capped.
  // consumer[s,a].start ≥ perSlabSubEndForSlab[s] (all per-slab subs complete)
  // and ≥ any per-atom RS sub-phase for this atom.
  // Plus ≥ previous consumer phase end (strict ordering).
  //
  // We emit RS ldmatrix-A per-atom producer sub-phases inline, tying each to
  // the consumer phase it feeds.

  let cCursor = 0;
  // First consumer can start no earlier than slab 0's per-slab subs complete.
  const firstSlabReady = perSlabSubEndForSlab[0] ?? 0;
  cCursor = firstSlabReady;

  let flatK = 0;
  outerSlab: for (let s = 0; s < producerSlabsDisplayed; s++) {
    for (let a = 0; a < atomsPerStage_K; a++) {
      if (flatK >= consumerItersDisplayed) break outerSlab;

      // Determine consumer start tick.
      const slabReady = perSlabSubEndForSlab[s] ?? 0;
      let start = Math.max(cCursor, slabReady);

      // Hoist per-atom producer sub-phases (RS ldmatrix) RIGHT BEFORE this
      // consumer phase. The sub-phase must have finished by `start`, so we
      // push `start` forward if needed to make room.
      for (const sub of perAtomSubs) {
        const subStart = start;
        const subEnd = subStart + sub.ticks;
        const stage = s % kStages;
        pushPhase(producer, {
          id: `${idPrefix}.${sub.kind}.prod.s${s}.a${a}`,
          kind: sub.kind,
          startTick: subStart,
          endTick: subEnd,
          label: sub.label(s, a),
          description: sub.description(s, a),
          iter: s,
          stage: sub.stageOverride ? sub.stageOverride(s) : stage,
          kSlab: s,
          kAtomInSlab: a,
          producerSub: sub.producerSub,
        });
        start = subEnd;
      }

      const end = start + ctx.mmaPhaseTicks;
      const stage = s % kStages;
      pushPhase(consumer, {
        id: `${idPrefix}.${ctx.consumerKind}.cons.s${s}.a${a}`,
        kind: ctx.consumerKind,
        startTick: start,
        endTick: end,
        label: consumerLabel(ctx, s, a, flatK, consumerItersTotal),
        description: consumerDescription(ctx, s, a),
        iter: flatK,
        stage,
        kSlab: s,
        kAtomInSlab: a,
      });
      cCursor = end;
      flatK += 1;
    }
  }

  // Tail collapse for consumer.
  if (consumerItersTail > 0) {
    // Duration scales with atomsPerStage_MN to represent the per-atom budget of
    // the collapsed iters. One bar; interpolates as a "× N more" marker.
    const tailTicks = ctx.mmaPhaseTicks;
    const start = cCursor;
    const end = start + tailTicks;
    // collapse phase carries consumerItersDisplayed as iter so the world can
    // still look up a plausible consumerAtom inside it.
    const collapsedSlab = Math.max(0, producerSlabsDisplayed);
    const collapsedAtom = 0;
    pushPhase(consumer, {
      id: `${idPrefix}.${ctx.consumerKind}.cons.tail`,
      kind: ctx.consumerKind,
      startTick: start,
      endTick: end,
      label: `× ${consumerItersTail} more mma`,
      description: `${consumerItersTail} additional mma steps collapsed; each consumes one atom-K slice of its stage; acc lands in ${accInLabel(i)}.`,
      iter: consumerItersDisplayed,
      stage: collapsedSlab % kStages,
      kSlab: collapsedSlab,
      kAtomInSlab: collapsedAtom,
      collapsedCount: consumerItersTail,
    });
    cCursor = end;
  }

  // --- Back-pressure fixup for producer under ring-full (plan §N invariant 7).
  // Slab s's FIRST per-slab sub-phase must start ≥ consumer[s-kStages, last
  // atom].end. If it doesn't, push slab s's subs forward, and cascade to any
  // later producer phases that now overlap. We skip collapsed tail phase.
  //
  // Implementation: index per-slab subs by slab-number; find the slab's first
  // sub-phase and its contiguous run; shift by the needed delta.
  if (producerSlabsDisplayed > kStages) {
    // Map slab → [startIdx..endIdx] of producer phases (per-slab only, not
    // per-atom RS subs).
    const slabBlocks: { start: number; end: number }[] = [];
    for (let s = 0; s < producerSlabsDisplayed; s++) {
      let startIdx = -1;
      let endIdx = -1;
      for (let i2 = 0; i2 < producer.length; i2++) {
        const p = producer[i2];
        if (p.kSlab !== s) continue;
        if (p.kAtomInSlab !== undefined) continue; // skip per-atom sub
        if (p.collapsedCount) continue;
        if (startIdx === -1) startIdx = i2;
        endIdx = i2;
      }
      slabBlocks.push({ start: startIdx, end: endIdx });
    }

    for (let s = kStages; s < producerSlabsDisplayed; s++) {
      const consumerIdxThatFreed = (s - kStages) * atomsPerStage_K + (atomsPerStage_K - 1);
      if (consumerIdxThatFreed >= consumer.length) continue;
      const cEnd = consumer[consumerIdxThatFreed].endTick;
      const block = slabBlocks[s];
      if (block.start === -1) continue;
      const curStart = producer[block.start].startTick;
      if (curStart < cEnd) {
        const delta = cEnd - curStart;
        // Shift this slab's per-slab block.
        for (let i3 = block.start; i3 <= block.end; i3++) {
          producer[i3].startTick += delta;
          producer[i3].endTick += delta;
        }
        // Cascade: any later per-slab block that overlaps now also shifts.
        for (let t = s + 1; t < producerSlabsDisplayed; t++) {
          const b = slabBlocks[t];
          if (b.start === -1) continue;
          const prevBlock = slabBlocks[t - 1];
          const prevEnd = prevBlock.end >= 0 ? producer[prevBlock.end].endTick : 0;
          if (producer[b.start].startTick < prevEnd) {
            const d = prevEnd - producer[b.start].startTick;
            for (let i4 = b.start; i4 <= b.end; i4++) {
              producer[i4].startTick += d;
              producer[i4].endTick += d;
            }
          }
        }
      }
    }
  }

  // --- Epilogue.
  const mainloopEnd = Math.max(
    producer.at(-1)?.endTick ?? 0,
    consumer.at(-1)?.endTick ?? 0,
  );
  let eCursor = mainloopEnd;

  if (i.accIn === 'tmem') {
    pushPhase(epilogue, {
      id: `${idPrefix}.tcgen05.ld`,
      kind: 'tcgen05.ld',
      startTick: eCursor,
      endTick: eCursor + TICKS.TCG_LD,
      label: 'tcgen05.ld → .reg',
      description:
        'Move accumulator from TMEM into per-lane registers for epilogue (16dp×N shapes; lanes fan out across subpartitions).',
    });
    eCursor += TICKS.TCG_LD;
  }

  const stgTicks = Math.max(TICKS.STG_SMEM, Math.ceil((input.blkMMult * input.blkNMult) / 4));
  pushPhase(epilogue, {
    id: `${idPrefix}.epilogue.stg_smem`,
    kind: 'epilogue.stg_smem',
    startTick: eCursor,
    endTick: eCursor + stgTicks,
    label: 'acc → SMEM staging',
    description:
      'Warp writes its accumulator fragments into an epilogue SMEM region (stmatrix). Fills left-to-right.',
  });
  eCursor += stgTicks;

  pushPhase(epilogue, {
    id: `${idPrefix}.epilogue.tma.store`,
    kind: 'epilogue.tma.store',
    startTick: eCursor,
    endTick: eCursor + TICKS.TMA_STORE,
    label: 'SMEM → GMEM (TMA store)',
    description:
      'cp.async.bulk.tensor.store drains the staging region to GMEM. Cells empty right-to-left.',
  });
  eCursor += TICKS.TMA_STORE;

  return {
    streams: { producer, consumer, epilogue },
    totalTicks: eCursor,
  };
}

function emitCoupled(ctx: EmitCtx): { streams: SimStreams; totalTicks: number } {
  const { summary, input } = ctx;
  const { slabCount, atomsPerStage_K } = summary;
  const consumerItersTotal = slabCount * atomsPerStage_K;
  const i = input.inst;
  const idPrefix = ctx.idPrefix;

  const producer: TimelinePhase[] = [];
  const consumer: TimelinePhase[] = [];
  const epilogue: TimelinePhase[] = [];

  const subs = describeProducerSubs(ctx);
  // For coupled, the producer stream interleaves with consumer to form a per-
  // k-step serial sequence. Both producer and consumer end up with slabCount×
  // atomsPerStage_K phases.
  const perSlabSubs = subs.filter((s) => s.perSlab);
  const perAtomSubs = subs.filter((s) => !s.perSlab);

  const consumerItersDisplayed = Math.min(consumerItersTotal, MAX_PHASES_PER_STREAM);
  const consumerItersTail = consumerItersTotal - consumerItersDisplayed;

  let t = 0;
  let flatK = 0;
  outer: for (let s = 0; s < slabCount; s++) {
    // Once per slab: emit the perSlab sub(s) (e.g., cp.async or wmma.load).
    for (const sub of perSlabSubs) {
      if (flatK >= consumerItersDisplayed) break outer;
      const start = t;
      const end = start + sub.ticks;
      pushPhase(producer, {
        id: `${idPrefix}.${sub.kind}.prod.s${s}`,
        kind: sub.kind,
        startTick: start,
        endTick: end,
        label: sub.label(s),
        description: sub.description(s),
        iter: s,
        stage: sub.stageOverride ? sub.stageOverride(s) : 0,
        kSlab: s,
        producerSub: sub.producerSub,
      });
      t = end;
    }

    // Per k-atom: for coupled wmma we mirror the wmma.load once per atom
    // (coupled, per-step pattern — slab-level loading ≠ buffering). For mma
    // we emit the ldmatrix + mma inline. To hit the memo contract
    // "producer.length === consumer.length === slabCount × atomsPerStage_K"
    // for wmma, duplicate the wmma.load as a per-atom sub-phase. For mma,
    // rely on the real ldmatrix sub-phase.
    for (let a = 0; a < atomsPerStage_K; a++) {
      if (flatK >= consumerItersDisplayed) break outer;

      // For wmma, the "producer" for each k-atom is the already-issued
      // wmma.load; we still emit a synthetic per-atom producer phase so the
      // 1:1 coupled pattern holds. For mma we emit ldmatrix.
      if (i.family === 'wmma') {
        // wmma: emit a short per-atom placeholder so producer.length ===
        // consumer.length. Per plan §K1, "producer emits per-k-step" in
        // coupled mode.
        if (a > 0) {
          // For the first atom, slab's wmma.load already suffices; emit an
          // atom-scoped copy so each k-atom has a matching producer.
          const start = t;
          const end = start + TICKS.WMMA_LOAD;
          pushPhase(producer, {
            id: `${idPrefix}.wmma.load.prod.s${s}.a${a}`,
            kind: 'wmma.load',
            startTick: start,
            endTick: end,
            label: `wmma.load slab ${s} · k-atom ${a}`,
            description: `Warp-wide wmma.load for atom-K slice ${a} of slab ${s}.`,
            iter: flatK,
            stage: 0,
            kSlab: s,
            kAtomInSlab: a,
            producerSub: 'wmma-load',
          });
          t = end;
        } else {
          // First atom's producer is the slab's wmma.load we already emitted.
          // Attach kAtomInSlab=0 to that last phase.
          const last = producer.at(-1);
          if (last && last.kSlab === s) {
            last.kAtomInSlab = 0;
          }
        }
      } else {
        // mma (sm_80/89): emit ldmatrix per k-atom.
        for (const sub of perAtomSubs) {
          const start = t;
          const end = start + sub.ticks;
          pushPhase(producer, {
            id: `${idPrefix}.${sub.kind}.prod.s${s}.a${a}`,
            kind: sub.kind,
            startTick: start,
            endTick: end,
            label: sub.label(s, a),
            description: sub.description(s, a),
            iter: flatK,
            stage: sub.stageOverride ? sub.stageOverride(s) : 0,
            kSlab: s,
            kAtomInSlab: a,
            producerSub: sub.producerSub,
          });
          t = end;
        }
      }

      const start = t;
      const end = start + ctx.mmaPhaseTicks;
      pushPhase(consumer, {
        id: `${idPrefix}.${ctx.consumerKind}.cons.s${s}.a${a}`,
        kind: ctx.consumerKind,
        startTick: start,
        endTick: end,
        label: consumerLabel(ctx, s, a, flatK, consumerItersTotal),
        description: consumerDescription(ctx, s, a),
        iter: flatK,
        stage: 0,
        kSlab: s,
        kAtomInSlab: a,
      });
      t = end;
      flatK += 1;
    }
  }

  // Tail collapse (coupled). Both streams pick up a "× N more" bar.
  if (consumerItersTail > 0) {
    const producerSubBudget =
      perSlabSubs.reduce((acc, s) => acc + s.ticks, 0) / Math.max(1, atomsPerStage_K) +
      perAtomSubs.reduce((acc, s) => acc + s.ticks, 0);
    const pTailTicks = Math.max(1, Math.ceil(producerSubBudget));
    const pStart = t;
    const pEnd = pStart + pTailTicks;
    const pKind: ProducerKind = i.family === 'wmma' ? 'wmma.load' : 'cp.async';
    const pSub: NonNullable<TimelinePhase['producerSub']> =
      i.family === 'wmma' ? 'wmma-load' : 'cpasync';
    pushPhase(producer, {
      id: `${ctx.idPrefix}.prod.tail`,
      kind: pKind,
      startTick: pStart,
      endTick: pEnd,
      label: `× ${consumerItersTail} more`,
      description: `${consumerItersTail} additional load-mma pairs collapsed.`,
      iter: consumerItersDisplayed,
      stage: 0,
      kSlab: Math.floor(consumerItersDisplayed / Math.max(1, atomsPerStage_K)),
      kAtomInSlab: consumerItersDisplayed % Math.max(1, atomsPerStage_K),
      collapsedCount: consumerItersTail,
      producerSub: pSub,
    });
    t = pEnd;

    const cStart = t;
    const cEnd = cStart + ctx.mmaPhaseTicks;
    pushPhase(consumer, {
      id: `${ctx.idPrefix}.${ctx.consumerKind}.cons.tail`,
      kind: ctx.consumerKind,
      startTick: cStart,
      endTick: cEnd,
      label: `× ${consumerItersTail} more mma`,
      description: `${consumerItersTail} additional mma iterations collapsed.`,
      iter: consumerItersDisplayed,
      stage: 0,
      kSlab: Math.floor(consumerItersDisplayed / Math.max(1, atomsPerStage_K)),
      kAtomInSlab: consumerItersDisplayed % Math.max(1, atomsPerStage_K),
      collapsedCount: consumerItersTail,
    });
    t = cEnd;
  }

  // Epilogue: single st.global / wmma.store step.
  const mainloopEnd = Math.max(
    producer.at(-1)?.endTick ?? 0,
    consumer.at(-1)?.endTick ?? 0,
  );
  let eCursor = mainloopEnd;
  const label = i.family === 'wmma'
    ? 'wmma.store_matrix_sync → GMEM (C)'
    : '.reg → GMEM (st.global)';
  const desc = i.family === 'wmma'
    ? 'Warp-wide wmma.store_matrix_sync writes the C fragment to gmem or shared directly; no SMEM staging.'
    : 'Each thread writes its mma accumulator fragment with plain st.global / stg; no SMEM staging.';
  pushPhase(epilogue, {
    id: `${ctx.idPrefix}.epilogue.stg_smem`,
    kind: 'epilogue.stg_smem',
    startTick: eCursor,
    endTick: eCursor + TICKS.ST_GLOBAL,
    label,
    description: desc,
  });
  eCursor += TICKS.ST_GLOBAL;

  return {
    streams: { producer, consumer, epilogue },
    totalTicks: eCursor,
  };
}

function emitPhases(input: SimInput, summary: SimSummary): { streams: SimStreams; totalTicks: number } {
  const i = input.inst;
  const ctx: EmitCtx = {
    input,
    summary,
    consumerKind: mmaStepKind(i),
    mmaPhaseTicks: TICKS.MMA * summary.atomsPerStage_MN,
    idPrefix: i.id,
  };

  if (summary.familyShape === 'tma-warpspec') {
    return emitWarpspec(ctx);
  }
  return emitCoupled(ctx);
}

// ---------------------------------------------------------------------------
// Part R step 4 — buildEffects
//
// For Phase 1 we only need effects that drive `active.*`, `progress.*`,
// `ring[]` roles and `mbar[]` state transitions. Richer effects arrive
// with later phases.
// ---------------------------------------------------------------------------

type PhaseStream = 'producer' | 'consumer' | 'epilogue';

function phaseEffect(stream: PhaseStream, phase: TimelinePhase): { start: Effect; end: Effect } {
  const start: Effect = {
    atTick: phase.startTick,
    apply: (s) => {
      s.active[stream] = phase;
    },
  };
  const end: Effect = {
    atTick: phase.endTick,
    apply: (s) => {
      // Only clear if this is still the active phase on this stream;
      // overlapping phases on the same stream never happen by construction,
      // but we guard just in case.
      if (s.active[stream]?.id === phase.id) s.active[stream] = null;
    },
  };
  return { start, end };
}

// Map a producer sub-kind to the operand it transfers. Plan §C4 / variant
// matrix §K data-transfer shapes.
function operandForProducerSub(
  sub: NonNullable<TimelinePhase['producerSub']>,
): ProducerTransferState['operand'] {
  switch (sub) {
    case 'tma':
    case 'cpasync':
    case 'wmma-load':
      return 'AB';
    case 'ldmatrixA':
    case 'tcgen05-cp':
      return 'A';
    case 'metadata':
      return 'meta';
    case 'scale':
      return 'scaleA';
  }
}

function linesTotalForProducerPhase(
  sub: NonNullable<TimelinePhase['producerSub']>,
  summary: SimSummary,
): number {
  switch (sub) {
    case 'tma':
    case 'cpasync':
    case 'wmma-load':
      return summary.linesPerSlab_AB;
    case 'ldmatrixA':
      return summary.linesPerAtom_ldmatrixA;
    case 'tcgen05-cp':
      return summary.linesPerSlab_tcgen05cp;
    case 'metadata':
      return summary.linesPerSlab_metadata;
    case 'scale':
      return summary.linesPerSlab_scale;
  }
}

function buildEffects(
  streams: SimStreams,
  summary: SimSummary,
  kStages: number,
  input: SimInput,
): Effect[] {
  const effects: Effect[] = [];

  // Active-phase effects for all three streams.
  for (const p of streams.producer) {
    const { start, end } = phaseEffect('producer', p);
    effects.push(start, end);
  }
  for (const p of streams.consumer) {
    const { start, end } = phaseEffect('consumer', p);
    effects.push(start, end);
  }
  for (const p of streams.epilogue) {
    const { start, end } = phaseEffect('epilogue', p);
    effects.push(start, end);
  }

  // ---------- C-tile accumulation (Phase 5, plan §C4, §N invariant 7) -----
  // Each non-collapsed consumer (mma step) phase fires on ALL (m, n) atoms
  // once: `cTile.accumulated[m][n] += 1` for every (m, n) at phase.end.
  // A COLLAPSE consumer phase represents `collapsedCount` folded consumer
  // phases; at its .end increment by `collapsedCount` for every (m, n).
  // (See §O5 — the per-atom indices inside a collapse bar lock to (0, 0),
  // but the accumulator must still advance by the folded count.)
  const bm = Math.max(1, input.blkMMult);
  const bn = Math.max(1, input.blkNMult);
  const cap = summary.maxAccumulatedPerAtom;
  for (const p of streams.consumer) {
    const isMmaStep = p.kind === 'wgmma.step' || p.kind === 'tcgen05.mma.step';
    if (!isMmaStep) continue;
    const bump = p.collapsedCount && p.collapsedCount > 0 ? p.collapsedCount : 1;
    effects.push({
      atTick: p.endTick,
      apply: (s) => {
        const t = s.cTile;
        if (!t) return;
        for (let m = 0; m < bm; m++) {
          for (let n = 0; n < bn; n++) {
            const next = t.accumulated[m][n] + bump;
            // Cap at maxAccumulatedPerAtom per invariant 7 (monotone,
            // bounded). Guards against any hypothetical over-emission.
            t.accumulated[m][n] = Math.min(cap, next);
          }
        }
      },
    });
  }

  // ---------- Epilogue sweep .end effects (Phase 5, plan §C4) -------------
  // When an epilogue staging/drain phase ends, ALL (m, n) cells reach 1
  // (invariant 9, monotone non-decreasing; sweeps clamp at 1). The mid-phase
  // interpolation in makeWorldAt handles the row-major 0→1 ramp.
  for (const p of streams.epilogue) {
    if (p.kind === 'epilogue.stg_smem') {
      effects.push({
        atTick: p.endTick,
        apply: (s) => {
          const t = s.cTile;
          if (!t) return;
          for (let m = 0; m < bm; m++) {
            for (let n = 0; n < bn; n++) t.epilogueStaged[m][n] = 1;
          }
        },
      });
    } else if (p.kind === 'epilogue.tma.store') {
      effects.push({
        atTick: p.endTick,
        apply: (s) => {
          const t = s.cTile;
          if (!t) return;
          for (let m = 0; m < bm; m++) {
            for (let n = 0; n < bn; n++) t.epilogueDrained[m][n] = 1;
          }
        },
      });
    }
  }

  // Ring + mbar transitions (warpspec only).
  //
  // Only the producer sub-phase that actually fills the slab (`tma`) drives
  // the ring into 'fill' → 'hold'. Ancillary sub-phases (ldmatrixA, tcgen05-cp,
  // metadata, scale) do NOT mutate the ring; they happen within the stage
  // already held by that slab.
  //
  // Only the LAST k-atom of a slab drains the ring to 'empty' + mbar empty
  // (plan §K3 and invariant 5 for slice consistency).
  if (summary.hasRing) {
    for (const p of streams.producer) {
      if (p.stage === undefined) continue;
      if (p.producerSub !== 'tma') continue; // only TMA fills
      const stage = p.stage;
      const slice = p.kSlab ?? p.iter ?? 0;
      effects.push({
        atTick: p.startTick,
        apply: (s) => {
          const r = s.ring[stage];
          if (r) {
            r.role = 'fill';
            r.slice = slice;
            r.fillFrac = 0;
          }
        },
      });
      effects.push({
        atTick: p.endTick,
        apply: (s) => {
          const r = s.ring[stage];
          if (r) {
            r.role = 'hold';
            r.fillFrac = 1;
          }
          const m = s.mbar[stage];
          if (m) {
            m.state = 'full';
            m.lastArriveTick = p.endTick;
          }
        },
      });
    }
    const atomsPerStage_K = summary.atomsPerStage_K;
    for (const p of streams.consumer) {
      if (p.stage === undefined) continue;
      const stage = p.stage;
      const slice = p.kSlab ?? p.iter ?? 0;
      const atom = p.kAtomInSlab ?? 0;
      const isLastAtom = atom === atomsPerStage_K - 1 || !!p.collapsedCount;

      effects.push({
        atTick: p.startTick,
        apply: (s) => {
          const r = s.ring[stage];
          if (r) {
            r.role = 'consume';
            r.slice = slice;
            // Stage was 'hold' with a full buffer; make explicit for panels.
            r.fillFrac = 1;
          }
        },
      });

      if (isLastAtom) {
        effects.push({
          atTick: p.endTick,
          apply: (s) => {
            const r = s.ring[stage];
            if (r) {
              r.role = 'empty';
              r.fillFrac = 0;
            }
            const m = s.mbar[stage];
            if (m) {
              m.state = 'empty';
              m.lastWaitTick = p.endTick;
            }
          },
        });
      }
      // Intermediate k-atoms leave the stage in 'consume' until the next
      // producer fills it or the last k-atom drains it.
    }
  }

  // Stable sort by atTick. `.sort` in v8 is already stable, but be explicit
  // by appending a monotonic index so phase-start effects fire before their
  // same-tick siblings.
  effects.sort((a, b) => a.atTick - b.atTick);
  return effects;
  // Note: kStages is a parameter for future use (e.g. seeding ring slices
  // before the first TMA load); unused here but kept for API stability.
  void kStages;
}

// ---------------------------------------------------------------------------
// Part R step 5 — worldAt
// ---------------------------------------------------------------------------

// Phase 6: warp count per family (plan §L1). wgmma/tcgen05(.block_scaled) run
// as a cooperative 4-warp warpgroup; mma/wmma run a single warp.
export function warpsForFamily(family: InstSpec['family']): number {
  if (family === 'wgmma' || family === 'tcgen05' || family === 'tcgen05.block_scaled') return 4;
  return 1;
}

function initialWorld(input: SimInput, summary: SimSummary): WorldState {
  const ring: RingSlotState[] = [];
  if (summary.familyShape === 'tma-warpspec') {
    for (let s = 0; s < input.kStages; s++) {
      ring.push({ stage: s, slice: s, role: 'empty', fillFrac: 0 });
    }
  } else if (summary.familyShape === 'cpasync-mma') {
    // Synthetic single-slot SMEM stage so SmemPanel has something to show.
    ring.push({ stage: 0, slice: 0, role: 'empty', fillFrac: 0 });
  }
  // wmma: ring.length === 0 (no SMEM at all).

  const mbar: MbarState[] = summary.hasMbar
    ? Array.from({ length: input.kStages }, (_, i) => ({
        stage: i,
        state: 'empty' as const,
        lastArriveTick: -1,
        lastWaitTick: -1,
      }))
    : [];

  // Phase 5: cTile is a 2-D grid shape [blkMMult][blkNMult] initialized to 0.
  // `accumulated[m][n]` counts how many consumer phases have fired on atom
  // (m, n); it caps at summary.maxAccumulatedPerAtom (= slabCount ×
  // atomsPerStage_K). `epilogueStaged` / `epilogueDrained` each sweep 0..1
  // per (m, n) in row-major order across their epilogue sub-phases.
  const bm = Math.max(1, input.blkMMult);
  const bn = Math.max(1, input.blkNMult);
  const mkGrid = () => {
    const g: number[][] = [];
    for (let m = 0; m < bm; m++) g.push(new Array(bn).fill(0));
    return g;
  };
  const cTile: CTileState = {
    accumulated: mkGrid(),
    epilogueStaged: mkGrid(),
    epilogueDrained: mkGrid(),
  };

  // Phase 6: seed per-warp records as 'idle' with empty fragments. The
  // dynamic role assignment happens per-tick in makeWorldAt (plan §L2).
  const nWarps = warpsForFamily(input.inst.family);
  const warps: WarpState[] = Array.from({ length: nWarps }, (_, idx) => ({
    warpIdx: idx,
    role: 'idle',
    fragment: { kind: 'none' },
  }));

  // Phase 6: cluster is non-null only for cg2 tcgen05 variants (plan §K6).
  // peerActive / sharedLoad are filled per-tick in makeWorldAt.
  const cluster: ClusterState | null =
    input.inst.ctaGroup === 2
      ? { thisCtaRole: 'leader', peerActive: false, sharedLoad: false }
      : null;

  return {
    tick: 0,
    active: { producer: null, consumer: null, epilogue: null },
    progress: { producer: 0, consumer: 0, epilogue: 0 },
    ring,
    mbar,
    // populated in phase 3
    producerTransfer: null,
    // populated in phase 4
    consumerAtom: null,
    // populated in phase 5
    cTile,
    // phase 6: per-warp roles + cluster + auxiliary
    warps,
    cluster,
    auxiliary: {
      metadata: !!input.inst.sparse,
      scaleA: input.inst.family === 'tcgen05.block_scaled',
      scaleB: input.inst.family === 'tcgen05.block_scaled',
    },
  };
}

function cloneWorld(w: WorldState): WorldState {
  return {
    tick: w.tick,
    active: { ...w.active },
    progress: { ...w.progress },
    ring: w.ring.map((r) => ({ ...r })),
    mbar: w.mbar.map((m) => ({ ...m })),
    producerTransfer: w.producerTransfer ? { ...w.producerTransfer } : null,
    consumerAtom: w.consumerAtom ? { ...w.consumerAtom } : null,
    cTile: w.cTile
      ? {
          accumulated: w.cTile.accumulated.map((r) => r.slice()),
          epilogueStaged: w.cTile.epilogueStaged.map((r) => r.slice()),
          epilogueDrained: w.cTile.epilogueDrained.map((r) => r.slice()),
        }
      : null,
    warps: w.warps.map((x) => ({ ...x, fragment: { ...x.fragment } })),
    cluster: w.cluster ? { ...w.cluster } : null,
    auxiliary: { ...w.auxiliary },
  };
}

function progressOf(phase: TimelinePhase | null, tick: number): number {
  if (!phase) return 0;
  const denom = Math.max(1, phase.endTick - phase.startTick);
  const frac = (tick - phase.startTick) / denom;
  return Math.min(1, Math.max(0, frac));
}

function makeWorldAt(
  input: SimInput,
  summary: SimSummary,
  effects: Effect[],
): (tick: number) => WorldState {
  return (tick: number) => {
    const w = initialWorld(input, summary);
    w.tick = tick;
    // Apply every effect whose atTick ≤ tick. We don't binary-search here
    // because effect count is small (O(phases)); a linear scan is simpler
    // and keeps the behaviour obvious. Can be upgraded later.
    for (const e of effects) {
      if (e.atTick <= tick) {
        e.apply(w);
      } else {
        break; // effects are sorted
      }
    }

    // Interpolate within the currently-active phases.
    if (w.active.producer) {
      w.progress.producer = progressOf(w.active.producer, tick);
    }
    if (w.active.consumer) {
      w.progress.consumer = progressOf(w.active.consumer, tick);
    }
    if (w.active.epilogue) {
      w.progress.epilogue = progressOf(w.active.epilogue, tick);
    }

    // Ring fillFrac interpolation: while a stage is in 'fill', ramp its
    // fillFrac with the producer progress. Later phases will interpolate
    // other per-slot quantities; the immutable clone above ensures the
    // hook is safe to mutate.
    if (w.active.producer && w.active.producer.stage !== undefined) {
      const r = w.ring[w.active.producer.stage];
      if (r && r.role === 'fill') {
        r.fillFrac = w.progress.producer;
      }
    }

    // producerTransfer: populated only while a producer phase is in flight
    // (plan §B, §C4). Kind/operand derived from the phase's `producerSub`;
    // linesLoaded ramps 0 → linesTotal linearly across the phase.
    if (w.active.producer) {
      const p = w.active.producer;
      const sub = p.producerSub;
      if (sub) {
        const linesTotal = linesTotalForProducerPhase(sub, summary);
        const frac = w.progress.producer;
        const linesLoaded = Math.min(
          linesTotal,
          Math.floor(frac * linesTotal),
        );
        // Bytes-in-flight: 128 B per line. Panels (SmemBudget) use this
        // directly; tests can regress against it as well.
        const bytesInFlight = linesLoaded * 128;
        w.producerTransfer = {
          kind: sub,
          kSlab: p.kSlab ?? p.iter ?? 0,
          operand: operandForProducerSub(sub),
          // For sub-phases that don't occupy a ring stage (wmma), the phase
          // may still carry stage=0/undefined; we fall through to -1 so
          // panels can opt out.
          stage: p.stage ?? -1,
          linesLoaded,
          linesTotal,
          bytesInFlight,
        };
      }
    }
    // When the producer stream is idle, producerTransfer must be null — the
    // effect sweep leaves initialWorld's null untouched, so no reset needed.

    // consumerAtom: populated only while a consumer (mma-step) phase is in
    // flight (plan §B, §C4). The MN-atom sub-state is multiplexed within
    // each consumer phase in m-outer, n-inner order (plan §A). atomFlatIdx
    // ramps 0 → atomsPerStage_MN − 1 linearly across the phase; laneWave
    // cycles through maxWaysConsumer within each atom.
    if (w.active.consumer) {
      const p = w.active.consumer;
      const isMmaStep =
        p.kind === 'wgmma.step' || p.kind === 'tcgen05.mma.step';
      if (isMmaStep) {
        const atomsMN = Math.max(1, summary.atomsPerStage_MN);
        const blkN = Math.max(1, input.blkNMult);
        const maxWays = Math.max(1, summary.maxWaysConsumer);
        const frac = w.progress.consumer;
        // Inside a collapse phase the per-atom indices don't meaningfully
        // map to "real" iterations; lock to (0,0) as per Phase 4 spec.
        const isCollapse = !!p.collapsedCount;
        let subIdx = 0;
        let atomM = 0;
        let atomN = 0;
        if (!isCollapse) {
          subIdx = Math.min(atomsMN - 1, Math.floor(frac * atomsMN));
          // m-outer, n-inner:
          //   subIdx = atomM * blkNMult + atomN
          atomM = Math.floor(subIdx / blkN);
          atomN = subIdx % blkN;
        }
        const kSlab = p.kSlab ?? 0;
        const kAtomInSlab = p.kAtomInSlab ?? 0;
        const kStep = p.iter ?? kSlab * summary.atomsPerStage_K + kAtomInSlab;
        const stage = p.stage ?? 0;
        // laneWave: advances faster than atomFlatIdx; one "wave" per
        // maxWays * atomsMN across the phase. Stays 0 when maxWays === 1.
        const laneWave = maxWays > 1
          ? Math.floor(frac * atomsMN * maxWays) % maxWays
          : 0;
        w.consumerAtom = {
          kSlab,
          kStep,
          kAtomInSlab,
          stage,
          atomM,
          atomN,
          atomFlatIdx: subIdx,
          laneWave,
          maxWays,
        };
      }
    }
    // When the consumer stream is idle (warmup, between phases, epilogue, or
    // on a non-MMA consumer phase), consumerAtom stays null.

    // ---------- cTile mid-phase interpolation (Phase 5) ---------------------
    // The .end effect already bumped `accumulated` for any consumer phase
    // whose endTick ≤ tick. Mid-phase, we show a smooth ramp INSIDE the
    // currently active phase so the UI doesn't look like a staircase.
    //
    // Regular phase: each (m, n) bumps once when `atomFlatIdx` reaches its
    // flat index. Collapse phase: all (m, n) advance together by
    // `floor(frac × collapsedCount)`.
    if (w.active.consumer && w.cTile) {
      const p = w.active.consumer;
      const isMmaStep =
        p.kind === 'wgmma.step' || p.kind === 'tcgen05.mma.step';
      if (isMmaStep) {
        const frac = w.progress.consumer;
        const isCollapse = !!p.collapsedCount && p.collapsedCount > 0;
        const cTile = w.cTile;
        const bmLocal = Math.max(1, input.blkMMult);
        const bnLocal = Math.max(1, input.blkNMult);
        if (isCollapse) {
          const collapsed = p.collapsedCount!;
          const add = Math.min(collapsed, Math.floor(frac * collapsed));
          if (add > 0) {
            for (let m = 0; m < bmLocal; m++) {
              for (let n = 0; n < bnLocal; n++) {
                cTile.accumulated[m][n] = Math.min(
                  summary.maxAccumulatedPerAtom,
                  cTile.accumulated[m][n] + add,
                );
              }
            }
          }
        } else {
          // Regular phase: bump each (m, n) once its flat index has been
          // reached by atomFlatIdx. We use `floor(frac × atomsMN)` to derive
          // the running count, then set `accumulated[m][n] += 1` iff this
          // atom's flat index < running. This gives a per-atom stepped ramp
          // without double-counting when the phase end effect later fires.
          const atomsMN = Math.max(1, summary.atomsPerStage_MN);
          const running = Math.min(atomsMN, Math.floor(frac * atomsMN));
          if (running > 0) {
            for (let m = 0; m < bmLocal; m++) {
              for (let n = 0; n < bnLocal; n++) {
                const flat = m * bnLocal + n;
                if (flat < running) {
                  cTile.accumulated[m][n] = Math.min(
                    summary.maxAccumulatedPerAtom,
                    cTile.accumulated[m][n] + 1,
                  );
                }
              }
            }
          }
        }
      }
    }

    // ---------- Epilogue sweeps (Phase 5) ----------------------------------
    // During `epilogue.stg_smem`, `epilogueStaged[m][n]` sweeps 0→1 in row-
    // major order across the MN atoms. During `epilogue.tma.store`, same
    // pattern for `epilogueDrained`. Outside epilogue, both stay at whatever
    // the latest .end effect left them (0 before any epilogue, 1 after).
    if (w.active.epilogue && w.cTile) {
      const p = w.active.epilogue;
      const frac = w.progress.epilogue;
      const bmLocal = Math.max(1, input.blkMMult);
      const bnLocal = Math.max(1, input.blkNMult);
      const totalCells = bmLocal * bnLocal;
      const perCell = (flatIdx: number) =>
        Math.max(0, Math.min(1, frac * totalCells - flatIdx));
      if (p.kind === 'epilogue.stg_smem') {
        for (let m = 0; m < bmLocal; m++) {
          for (let n = 0; n < bnLocal; n++) {
            const flat = m * bnLocal + n;
            w.cTile.epilogueStaged[m][n] = Math.max(
              w.cTile.epilogueStaged[m][n],
              perCell(flat),
            );
          }
        }
      } else if (p.kind === 'epilogue.tma.store') {
        for (let m = 0; m < bmLocal; m++) {
          for (let n = 0; n < bnLocal; n++) {
            const flat = m * bnLocal + n;
            w.cTile.epilogueDrained[m][n] = Math.max(
              w.cTile.epilogueDrained[m][n],
              perCell(flat),
            );
          }
        }
      }
    }

    // ---------- Phase 6: per-warp role + cluster state (plan §L, §K6) ------
    // Roles refresh every tick from the active phases. We don't push these
    // through the effect list because they're a pure function of (active.*)
    // already reconstructed above, and tests expect role transitions on
    // worldAt() results without depending on effect stable-sort shenanigans.
    const hasProducer = !!w.active.producer;
    const hasConsumer = !!w.active.consumer;
    const hasEpilogue = !!w.active.epilogue;
    const isWs = !!input.inst.warpSpecialized;
    const fam = input.inst.family;
    const isCoupledWarpgroup =
      fam === 'wgmma' || fam === 'tcgen05' || fam === 'tcgen05.block_scaled';

    if (w.warps.length === 1) {
      // mma/wmma: single warp follows the most-active stream.
      const r: WarpState['role'] = hasConsumer
        ? 'consumer'
        : hasProducer
          ? 'producer'
          : hasEpilogue
            ? 'epilogue'
            : 'idle';
      w.warps[0].role = r;
    } else if (isCoupledWarpgroup && !isWs) {
      // Non-.ws warpgroup (wgmma, tcgen05 without warpSpecialized): all 4
      // warps share the same role at any tick, following the "most active"
      // stream. Consumer > epilogue > producer > idle (plan §L2).
      const r: WarpState['role'] = hasConsumer
        ? 'consumer'
        : hasEpilogue
          ? 'epilogue'
          : hasProducer
            ? 'producer'
            : 'idle';
      for (const wr of w.warps) wr.role = r;
    } else if (isWs) {
      // .ws subset (tcgen05 cg1 only, plan §K7 / §L3):
      //   warp 0 → producer during any producer phase, epilogue during
      //     epilogue.tma.store, else idle.
      //   warps 1..3 → consumer during any consumer phase, epilogue during
      //     epilogue.stg_smem, else idle.
      const epiKind = w.active.epilogue?.kind;
      // Warp 0 role
      let r0: WarpState['role'] = 'idle';
      if (hasEpilogue && epiKind === 'epilogue.tma.store') r0 = 'epilogue';
      else if (hasProducer) r0 = 'producer';
      w.warps[0].role = r0;
      // Warps 1..3 role
      let rC: WarpState['role'] = 'idle';
      if (hasEpilogue && epiKind === 'epilogue.stg_smem') rC = 'epilogue';
      else if (hasConsumer) rC = 'consumer';
      else if (hasEpilogue && epiKind === 'tcgen05.ld') rC = 'epilogue';
      for (let wi = 1; wi < w.warps.length; wi++) w.warps[wi].role = rC;
      // If both producer and consumer are active at different streams,
      // warp 0 stays 'producer' while 1..3 stay 'consumer' — the concurrent
      // case per plan §L3 steady state.
    }

    // Cluster (cg2 only): peerActive when ANY stream phase is active;
    // sharedLoad when the producer phase is a tma (multicast).
    if (w.cluster) {
      w.cluster.peerActive = hasProducer || hasConsumer || hasEpilogue;
      w.cluster.sharedLoad =
        hasProducer && w.active.producer?.producerSub === 'tma';
    }

    return w;
  };
}

// ---------------------------------------------------------------------------
// Part R step 6 — simulate
// ---------------------------------------------------------------------------

export function simulate(input: SimInput): SimResult {
  const summary = deriveSummary(input);
  const { streams, totalTicks } = emitPhases(input, summary);
  const effects = buildEffects(streams, summary, input.kStages, input);
  const worldAt = makeWorldAt(input, summary, effects);
  return {
    totalTicks,
    streams,
    effects,
    worldAt,
    summary,
  };
}

// Re-export cloneWorld so tests that want to snapshot a world can do so.
export { cloneWorld };
