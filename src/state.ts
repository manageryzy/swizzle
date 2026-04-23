// Single source of truth. Every panel derives from (spec, tick).

import { signal, computed } from '@preact/signals';
import {
  INSTRUCTIONS,
  type Dtype,
  type InstSpec,
  type Major,
  type OperandSource,
} from './instructions';
import { effectiveSwizzle, type SwizzleKind } from './swizzle';
import { PATTERNS, analyzeConflicts, type Access, type Conflict, type PatternContext } from './patterns';
import { tileDimsFor, type TileDims } from './tile_dims';
import { bytesOf } from './smem_budget';
import { type PipelineMode } from './pipeline_state';
import { simulate, type SimInput, type SimResult, type SimSummary, type WorldState } from './simulation';
import {
  DEFAULT_CONFIG_PER_MODE,
  modeOfInst,
  resolveInst,
  type Mode,
  type ModeConfig,
} from './inst_resolver';

export type ASource = OperandSource;

export interface Spec {
  instId: string;
  swizzle: SwizzleKind;
  majorA: Major;
  majorB: Major;
  aSource: ASource;
}

export const DEFAULT_SPEC: Spec = {
  instId: 'sm100.tcgen05.cg1.m64n128k16.f16',
  swizzle: '128B',
  majorA: 'K',
  majorB: 'K',
  aSource: 'smem',
};

export const spec = signal<Spec>(DEFAULT_SPEC);

// Dtype signals — carried independently from `spec` because one InstSpec can
// cover multiple dtypes (e.g. wgmma.m64n128k16.f16 allows both fp16 and bf16
// in aDtypes). ConfigBar tracks the user's explicit choice here.
export const dtypeA = signal<Dtype>('fp16');
export const dtypeB = signal<Dtype>('fp16');
export const accDtype = signal<Dtype>('fp32');

// Structured ModeConfig view over the current state. Read-only.
export const modeConfig = computed<ModeConfig>(() => {
  const i = inst.value;
  const s = spec.value;
  return {
    mode: modeOfInst(i),
    M: i.M, N: i.N, K: i.K,
    dtypeA: i.aDtypes.includes(dtypeA.value) ? dtypeA.value : i.aDtypes[0],
    dtypeB: i.bDtypes.includes(dtypeB.value) ? dtypeB.value : i.bDtypes[0],
    accDtype: i.accDtypes.includes(accDtype.value) ? accDtype.value : i.accDtypes[0],
    majorA: s.majorA, majorB: s.majorB,
    aSource: i.aSource.includes(s.aSource) ? s.aSource : i.aSource[0],
    ctaGroup: i.ctaGroup,
    sparse: i.sparse,
    warpSpecialized: i.warpSpecialized,
  };
});

// Apply a structured patch → resolve → write back to spec + dtype signals.
// Returns true on successful resolve; false when no catalog entry matches.
export function commitConfig(patch: Partial<ModeConfig>): boolean {
  const cur = modeConfig.value;
  const next: ModeConfig = { ...cur, ...patch };
  // RS (wgmma, rmem-source A) and TS (tcgen05, tmem-source A) both
  // static_assert A K-major in CUTLASS — see mma_sm90_gmma.hpp:190-191 and
  // mma_sm100_umma.hpp:134,178. Force the layout to satisfy the instruction
  // constraint rather than letting resolveInst silently fail.
  if (next.aSource === 'rmem' || next.aSource === 'tmem') {
    next.majorA = 'K';
  }
  const resolved = resolveInst(next);
  if (!resolved) return false;
  dtypeA.value = next.dtypeA;
  dtypeB.value = next.dtypeB;
  accDtype.value = next.accDtype;
  spec.value = {
    ...spec.value,
    instId: resolved.id,
    majorA: next.majorA,
    majorB: next.majorB,
    aSource: next.aSource ?? spec.value.aSource,
  };
  return true;
}

// Switch mode and pick that mode's canonical default config.
export function commitMode(mode: Mode): boolean {
  return commitConfig(DEFAULT_CONFIG_PER_MODE[mode]);
}
// Float tick — integer ticks are phase boundaries, the fractional part is
// sub-phase progress (used to drive smooth animations in the panels).
export const tick = signal<number>(0);
export const playing = signal<boolean>(false);
export const playbackRate = signal<number>(1); // ticks per second

// Clamp tick into the current instruction's valid range whenever the
// instruction id changes, so that switching kinds doesn't leave the cursor
// inside a phase that no longer exists.
let lastInstId = '';
export function resetTickIfInstChanged(newInstId: string) {
  if (lastInstId && lastInstId !== newInstId) {
    tick.value = 0;
    clampWarpSel();
  }
  lastInstId = newInstId;
}

// CUTLASS kStages — pipeline depth of the SMEM ring buffer. Each stage holds
// one (A,B) tile pair; iter X of the K mainloop consumes stage (X % kStages).
// Shared globally so the SMEM tiles, budget bar, and budget highlights all
// agree. "Stage axis" determines how multiple stages are visually stacked:
// along M (rows) or K (columns) of a combined canvas.
export const kStages = signal<number>(3);
export type StageAxis = 'vertical' | 'horizontal';
export const stageAxis = signal<StageAxis>('vertical');

// CTA-level tileK. Each MMA atom fires once per K iteration, where
// `numIters = ceil(tileK / i.K)` (see `kIterations` in smem_budget.ts). The
// SmemBudget panel writes this signal; the Timeline reads it to size the
// mma-step phase count. Default 4 atoms keeps the timeline readable when the
// panel is first opened.
const INITIAL_INST = INSTRUCTIONS.find((i) => i.id === DEFAULT_SPEC.instId) ?? INSTRUCTIONS[0];
export const tileK = signal<number>(INITIAL_INST.K * 4);

// When false (default), the wgmma N picker only lists the 8 N values that
// CUTLASS ships as canonical `SM90_64xNxK_*` atoms. Toggling this on
// exposes the remaining 24 PTX-valid N values (every other multiple of 8
// up to 256); those are labelled `[PTX-only]` in the dropdown.
export const wgmmaShowAllShapes = signal<boolean>(false);

// Warps per group — wgmma and tcgen05 run as a 4-warp warpgroup collective;
// sm_80 mma runs a single warp; wmma varies by atom. Keeping this computed
// here centralises the assumption every SIMT panel depends on.
export const warpsInGroup = computed<number>(() => {
  const fam = inst.value.family;
  if (fam === 'wgmma' || fam === 'tcgen05' || fam === 'tcgen05.block_scaled') return 4;
  return 1;
});

// SIMT selection state — linked across every panel that shows per-lane
// detail. `warpSel` picks which warp in the warpgroup we are inspecting;
// `laneSel` picks one of the 32 lanes (null = no lane focused); `bankSel`
// mirrors the same idea for the 32 SMEM banks; `cycleSel` is the replay
// cycle offset inside the currently-shown bank-conflict wave.
export const warpSel = signal<number>(0);
export const laneSel = signal<number | null>(null);
export const bankSel = signal<number | null>(null);
export const cycleSel = signal<number>(0);

// Information density — compact (default) hides secondary annotations so
// the whole page fits on a laptop; detail mode widens cells, enables inline
// labels, and surfaces every legend. Panels listen to this via CSS
// attribute selectors (see `[data-density]` in style.css).
export type DensityMode = 'compact' | 'detail';
export const densityMode = signal<DensityMode>('compact');

// CUTLASS TileShape multipliers. `inst.M/N/K` is the MMA *atom* shape. A real
// CUTLASS kernel tiles many atoms into one CTA via `TiledMMA<AtomLayoutMNK>`.
// blkMMult=blkNMult=1 (default) → one atom per CTA (the simplest case the
// original demo assumed). Larger values replicate the atom along M/N inside
// the CTA, growing `BLK_M/BLK_N` proportionally.
export const blkMMult = signal<number>(1);
export const blkNMult = signal<number>(1);

// Resolved CTA tile shape. `blkK` is the CTA K-tile (the CUTLASS `BLK_K`);
// we already had it under the name `tileK` from v1, kept for URL back-compat.
export const blkM = computed<number>(() => inst.value.M * blkMMult.value);
export const blkN = computed<number>(() => inst.value.N * blkNMult.value);
export const blkK = computed<number>(() => tileK.value);

// How many MMA atoms tile the CTA in (M, N). In CUTLASS this is
// `AtomLayoutMNK` of the `TiledMMA`.
export const atomsPerCta = computed<{ m: number; n: number }>(() => ({
  m: blkMMult.value,
  n: blkNMult.value,
}));

// Display-only ClusterShape. sm_90/100 kernels pack CTAs into clusters; the
// `ctaGroup` on tcgen05 instructions pairs two CTAs (ClusterShape = (2,1,1));
// everything else is (1,1,1). No hardware model, just the tuple.
export const clusterShape = computed<[number, number, number]>(() => {
  const cg = inst.value.ctaGroup ?? 1;
  return [cg, 1, 1];
});

// GMEM problem shape as multiples of the CTA tile. The visualizer walks one
// CTA's slice of the problem; the multiples control how many CTA tiles fit in
// each dimension (i.e. the ctaGrid extent). Default 4× on every axis gives a
// 4×4 CTA grid on the M×N face with 4 K slabs — legible without overwhelming.
export const problemMMult = signal<number>(4);
export const problemNMult = signal<number>(4);
export const problemKMult = signal<number>(4);

// Resolved GMEM problem dimensions (elements).
export const problemM = computed<number>(() => blkM.value * problemMMult.value);
export const problemN = computed<number>(() => blkN.value * problemNMult.value);
export const problemK = computed<number>(() => blkK.value * problemKMult.value);

// How many CTA tiles fit on each axis of the problem grid. ctaGrid.slicesK is
// the total number of K-slab loads (ignoring pipelining); each slab is one
// `tileK` worth of GMEM. Drives GmemPanel Track 1's stripe count.
export const ctaGrid = computed<{ rowsM: number; colsN: number; slicesK: number }>(() => ({
  rowsM: Math.max(1, Math.ceil(problemM.value / blkM.value)),
  colsN: Math.max(1, Math.ceil(problemN.value / blkN.value)),
  slicesK: Math.max(1, Math.ceil(problemK.value / blkK.value)),
}));

// Which CTA we're visualizing. For now fixed to (0, 0); leaving it computed
// makes it trivial to wire a user-chosen coordinate later.
export const thisCtaCoord = computed<{ m: number; n: number }>(() => ({ m: 0, n: 0 }));

// GMEM tensor geometry per operand. Respects majorA/majorB (K-major vs
// MN-major) so the row stride flips when the user toggles layout. These are
// the values GmemPanel uses to draw the GMEM rectangle and derive the
// line-by-line load pattern.
export interface GmemGeom {
  /** rows of GMEM rectangle */
  rows: number;
  /** cols of GMEM rectangle (one cell per element) */
  cols: number;
  /** bytes per row */
  rowStrideBytes: number;
  /** bytes per element */
  elemBytes: number;
  /** total bytes of the tensor */
  totalBytes: number;
  /** dtype label for the operand */
  dtypeLabel: string;
  /** which axis is the "fast" axis — useful for labels */
  major: Major;
}

function gmemGeomFor(
  operand: 'A' | 'B' | 'C',
  pM: number,
  pN: number,
  pK: number,
  i: InstSpec,
  majorA: Major,
  majorB: Major,
): GmemGeom {
  // C is always row-major (BLK_M × BLK_N in epilogue); we model it that way.
  const dtype =
    operand === 'A'
      ? i.aDtypes[0]
      : operand === 'B'
        ? i.bDtypes[0]
        : i.accDtypes[0];
  const eb = bytesOf(dtype);
  if (operand === 'A') {
    // A is (M × K). K-major → rows=M, cols=K. MN-major → rows=K, cols=M.
    const rows = majorA === 'K' ? pM : pK;
    const cols = majorA === 'K' ? pK : pM;
    const rowStrideBytes = Math.max(4, Math.ceil(cols * eb));
    return { rows, cols, rowStrideBytes, elemBytes: eb, totalBytes: rows * rowStrideBytes, dtypeLabel: dtype, major: majorA };
  }
  if (operand === 'B') {
    // B is (K × N). K-major → rows=N, cols=K (so N is outer and K is fast).
    // MN-major → rows=K, cols=N.
    const rows = majorB === 'K' ? pN : pK;
    const cols = majorB === 'K' ? pK : pN;
    const rowStrideBytes = Math.max(4, Math.ceil(cols * eb));
    return { rows, cols, rowStrideBytes, elemBytes: eb, totalBytes: rows * rowStrideBytes, dtypeLabel: dtype, major: majorB };
  }
  // C is (M × N), row-major in the epilogue staging.
  const rows = pM;
  const cols = pN;
  const rowStrideBytes = Math.max(4, Math.ceil(cols * eb));
  return { rows, cols, rowStrideBytes, elemBytes: eb, totalBytes: rows * rowStrideBytes, dtypeLabel: dtype, major: 'K' };
}

export const gmemA = computed<GmemGeom>(() =>
  gmemGeomFor('A', problemM.value, problemN.value, problemK.value, inst.value, spec.value.majorA, spec.value.majorB),
);
export const gmemB = computed<GmemGeom>(() =>
  gmemGeomFor('B', problemM.value, problemN.value, problemK.value, inst.value, spec.value.majorA, spec.value.majorB),
);
export const gmemC = computed<GmemGeom>(() =>
  gmemGeomFor('C', problemM.value, problemN.value, problemK.value, inst.value, spec.value.majorA, spec.value.majorB),
);

// Pipeline regime. Drives whether the Timeline and MemFlowPanel treat the
// producer and consumer as concurrent streams (warpspec) or as serial legs of
// a single warp (coupled). Sourced from the InstSpec at `pipelineMode`.
export const pipelineMode = computed<PipelineMode>(() => inst.value.pipelineMode);

// Keep `warpSel` inside the current instruction's warpgroup bounds whenever
// the instruction changes (e.g. switching from wgmma to mma must drop the
// selection from 3 to 0). Callers invoke this from `resetTickIfInstChanged`.
function clampWarpSel() {
  const w = warpsInGroup.value;
  if (warpSel.value >= w) warpSel.value = 0;
}

// Progress within the current (consumer-priority) phase, in [0, 1]. Panels
// that need producer-side progress (e.g. GmemPanel's load arrows) should
// derive from `currentProducerPhase` directly.
//
// Shim: world.value.progress.{consumer, epilogue, producer} is the source of
// truth. We pick the same precedence `currentPhase` uses (consumer > epilogue
// > producer) so existing callers see identical values.
export const phaseProgress = computed<number>(() => {
  const w = world.value;
  if (w.active.consumer) return w.progress.consumer;
  if (w.active.epilogue) return w.progress.epilogue;
  if (w.active.producer) return w.progress.producer;
  return 0;
});

// Progress within the current producer phase (tma.load / cp.async). Used by
// GmemPanel to sweep the GMEM→SMEM arrows top-to-bottom in time.
export const producerPhaseProgress = computed<number>(() => world.value.progress.producer);

// Progress within the current epilogue phase (tcgen05.ld / stg_smem / tma.store).
export const epiloguePhaseProgress = computed<number>(() => world.value.progress.epilogue);

export const inst = computed<InstSpec>(() => {
  const s = spec.value;
  return INSTRUCTIONS.find((i) => i.id === s.instId) ?? INSTRUCTIONS[0];
});

// Per-operand effective swizzle. The canonical cute atoms are defined for
// fp16 (2-byte elements, M=1 at byte level); for other element sizes the
// preserved region M shifts so the element boundary stays aligned. Callers
// that want the A-side view should read `activeSwizzleA`; mixed-precision
// kernels (e.g. fp8 × fp8 → fp32 acc) need `activeSwizzleB` too. Keeping
// `activeSwizzle` as an alias for the A-side keeps existing callers working
// without silently pinning them to fp16.
export const activeSwizzleA = computed(() =>
  effectiveSwizzle(spec.value.swizzle, bytesOf(dtypeA.value)),
);
export const activeSwizzleB = computed(() =>
  effectiveSwizzle(spec.value.swizzle, bytesOf(dtypeB.value)),
);
export const activeSwizzle = activeSwizzleA;

export const activePatternId = signal<string>('ldmatrix.x4.N');

// Which operand (A or B) the SMEM panel is currently displaying.
export const activeOperand = signal<'A' | 'B'>('A');

// The byte offset currently focussed in the bitfield panel; driven by
// SmemPanel hover.
export const focusedOffset = signal<number | null>(null);

// Tile dims of the currently-displayed operand.
export const activeTileDims = computed<TileDims>(() => {
  const i = inst.value;
  const op = activeOperand.value;
  const major = op === 'A' ? spec.value.majorA : spec.value.majorB;
  return tileDimsFor(i, op, major);
});

export const activePatternCtx = computed<PatternContext>(() => {
  const t = activeTileDims.value;
  return { rowStrideBytes: t.rowStrideBytes, tileBytes: t.tileBytes };
});
export const activePattern = computed(() => PATTERNS[activePatternId.value]);
export const currentAccesses = computed<Access[]>(() =>
  activePattern.value.accesses(activePatternCtx.value),
);
export const currentConflicts = computed<Conflict[]>(() =>
  analyzeConflicts(currentAccesses.value, activeSwizzle.value),
);
export const maxConflict = computed<number>(() => {
  const cs = currentConflicts.value;
  return cs.length === 0 ? 1 : cs[0].way;
});

export interface Phase {
  id: string;
  kind:
    | 'tma.load'
    | 'cp.async'
    | 'ldmatrix'
    | 'wmma.load'
    | 'tcgen05.cp'
    | 'metadata'
    | 'scale'
    | 'wgmma.step'
    | 'tcgen05.mma.step'
    | 'tcgen05.ld'
    | 'tcgen05.st'
    | 'epilogue.stg_smem'
    | 'epilogue.tma.store';
  startTick: number;
  endTick: number;
  label: string;
  description: string;
  /** K iteration this phase corresponds to (producer or consumer stream). */
  iter?: number;
  /** Ring stage slot this phase reads/writes. */
  stage?: number;
  /** K-slab index (producer: which TMA; consumer: which slab). */
  kSlab?: number;
  /** For consumer phases: atom-K slice within the slab. */
  kAtomInSlab?: number;
  /** Collapse phase marker: how many additional iters this covers. */
  collapsedCount?: number;
  /** Producer sub-phase classification (used by Phase 3 MemFlow wiring). */
  producerSub?: 'tma' | 'cpasync' | 'wmma-load' | 'ldmatrixA' | 'tcgen05-cp' | 'metadata' | 'scale';
}

// v4 simulator wiring — all of the per-tick state below is a thin view over
// `simResult` / `world`. See src/simulation.ts for the full SimInput →
// WorldState pipeline. Panels continue to read the same signal names as
// before; their internal shapes are preserved for source compatibility.
export const simInput = computed<SimInput>(() => ({
  inst: inst.value,
  majorA: spec.value.majorA,
  majorB: spec.value.majorB,
  swizzle: spec.value.swizzle,
  dtypeA: dtypeA.value,
  dtypeB: dtypeB.value,
  accDtype: accDtype.value,
  aSource: spec.value.aSource,
  blkMMult: blkMMult.value,
  blkNMult: blkNMult.value,
  tileK: tileK.value,
  kStages: kStages.value,
  problemMMult: problemMMult.value,
  problemNMult: problemNMult.value,
  problemKMult: problemKMult.value,
}));

export const simResult = computed<SimResult>(() => simulate(simInput.value));
export const summary = computed<SimSummary>(() => simResult.value.summary);
export const world = computed<WorldState>(() => simResult.value.worldAt(tick.value));

// Timeline emission — producer / consumer / epilogue phase streams. Under
// `warpspec` the producer and consumer may overlap in time; under `coupled`
// they collapse to a single serial stream. See `src/simulation.ts · simulate`.
export const timelineState = computed(() => {
  const r = simResult.value;
  return {
    producer: r.streams.producer,
    consumer: r.streams.consumer,
    epilogue: r.streams.epilogue,
    totalTicks: r.totalTicks,
  };
});

export const producerPhases = computed<Phase[]>(() => timelineState.value.producer);
export const consumerPhases = computed<Phase[]>(() => timelineState.value.consumer);
export const epiloguePhases = computed<Phase[]>(() => timelineState.value.epilogue);

// Flat union sorted by startTick — for panels that still iterate a single
// list (MemFlowPanel, SmemBudgetPanel, etc.). Prefer the per-stream computeds
// when you need overlap detection.
export const phases = computed<Phase[]>(() => {
  const s = timelineState.value;
  return [...s.producer, ...s.consumer, ...s.epilogue].sort((a, b) => a.startTick - b.startTick);
});

export const totalTicks = computed<number>(() => Math.max(1, timelineState.value.totalTicks));

// Find the most "visible" phase at the cursor: consumer > epilogue > producer.
// Most panels want the consumer-side event (they animate based on what the
// computation is doing). GmemPanel opts into producer directly via
// `currentProducerPhase` for its load arrows.
export const currentConsumerPhase = computed<Phase | null>(() => world.value.active.consumer);
export const currentProducerPhase = computed<Phase | null>(() => world.value.active.producer);
export const currentEpiloguePhase = computed<Phase | null>(() => world.value.active.epilogue);

export const currentPhase = computed<Phase | null>(() => {
  return currentConsumerPhase.value ?? currentEpiloguePhase.value ?? currentProducerPhase.value;
});

// mma.step phases often appear in sequence k=0, k=1, ... — this computes the
// k-index of the current mma step (so the SMEM panel can highlight one column
// stripe at a time).
//
// Shim: reads the active consumer phase out of `world`. Phase 4 will swap
// this to `world.value.consumerAtom?.kStep` once that field is populated.
export const currentKStep = computed<{ k: number; total: number } | null>(() => {
  const cur = currentConsumerPhase.value;
  if (!cur) return null;
  if (cur.kind !== 'wgmma.step' && cur.kind !== 'tcgen05.mma.step') return null;
  const steps = consumerPhases.value.filter((p) => p.kind === cur.kind);
  const k = steps.findIndex((p) => p.id === cur.id);
  return k >= 0 ? { k, total: steps.length } : null;
});

// Which K iter the producer is currently filling (warpspec only). In
// steady state this leads the consumer by `kStages − 1`; at the head it
// ramps up from 0 while consumer is stalled; at the tail it idles while
// the consumer drains. For coupled regime producer and consumer are on
// the same step, so this collapses to currentKStep.
//
// Shim: reads the active producer phase. Phase 3 will swap this to
// `world.value.producerTransfer?.kSlab` once that field is populated.
export const currentProducerKStep = computed<{ k: number; total: number } | null>(() => {
  const cur = currentProducerPhase.value;
  if (!cur) return null;
  if (cur.kind !== 'tma.load' && cur.kind !== 'cp.async') return null;
  const steps = producerPhases.value.filter((p) => p.kind === cur.kind);
  const k = steps.findIndex((p) => p.id === cur.id);
  return k >= 0 ? { k, total: steps.length } : null;
});
