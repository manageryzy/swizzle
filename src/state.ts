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
import { bytesOf, kIterations } from './smem_budget';
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

// Progress within the current phase, in [0, 1].
export const phaseProgress = computed<number>(() => {
  const t = tick.value;
  const p = phases.value.find((ph) => t >= ph.startTick && t < ph.endTick);
  if (!p) return 0;
  return (t - p.startTick) / Math.max(1, p.endTick - p.startTick);
});

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
}

// Maximum mma-step phases rendered in the Timeline. The K loop often has
// 16 / 32 atoms; drawing that many as individual phases makes the timeline
// illegible. We render up to MAX_MMA_PHASES head atoms, then emit a single
// "×N" collapse phase that represents the remaining iterations.
const MAX_MMA_PHASES = 8;

// Skeleton phase list — fleshed out in M1/M2/M3 per instruction family.
export function phasesFor(i: InstSpec, tileKValue: number): Phase[] {
  const base: Phase[] = [];
  let t = 0;
  const push = (p: Omit<Phase, 'startTick' | 'endTick' | 'id'> & { ticks: number }) => {
    base.push({
      id: `${i.id}.${p.kind}.${t}`,
      kind: p.kind,
      label: p.label,
      description: p.description,
      startTick: t,
      endTick: t + p.ticks,
    });
    t += p.ticks;
  };

  if (i.family === 'wgmma' || i.family === 'tcgen05' || i.family === 'tcgen05.block_scaled') {
    push({ kind: 'tma.load', label: 'TMA → SMEM (A+B into stage s)', description: 'cp.async.bulk.tensor loads A and B into the current pipeline stage; producer arrives on mbar.full.', ticks: 4 });
  } else {
    push({ kind: 'cp.async', label: 'cp.async → SMEM (A,B)', description: 'Per-thread cp.async copies; cp.async.commit + wait_group gate completion.', ticks: 4 });
  }

  if (i.family === 'mma') {
    push({ kind: 'ldmatrix', label: 'ldmatrix → .reg', description: 'Warp loads matrix fragment from swizzled SMEM into per-lane registers.', ticks: 2 });
  }

  const step = i.family === 'tcgen05' || i.family === 'tcgen05.block_scaled' ? 'tcgen05.mma.step' : 'wgmma.step';
  // K iterations follow the CUTLASS convention: numIters = tileK / kAtomK.
  // kStages (pipeline depth) is independent — each iter consumes stage
  // (iter % kStages). The caller's tileK drives the count; we render up to
  // MAX_MMA_PHASES iterations individually, then fold the rest into a single
  // collapse phase to keep the timeline legible.
  const numIters = kIterations(i, tileKValue);
  const headIters = Math.min(numIters, MAX_MMA_PHASES);
  for (let k = 0; k < headIters; k++) {
    const accIn = i.accIn === 'rmem' ? '.reg' : i.accIn.toUpperCase();
    push({
      kind: step,
      label: `mma iter ${k + 1}/${numIters}`,
      description: `Tensor core consumes K atom from stage (${k} mod kStages); accumulator in ${accIn}. Producer may already be filling the next stage.`,
      ticks: 2,
    });
  }
  const tailIters = numIters - headIters;
  if (tailIters > 0) {
    const accIn = i.accIn === 'rmem' ? '.reg' : i.accIn.toUpperCase();
    push({
      kind: step,
      label: `mma ×${tailIters} more`,
      description: `${tailIters} additional K iterations collapsed for legibility; each consumes one stage of the pipeline and lands in the ${accIn} accumulator.`,
      ticks: 2,
    });
  }

  if (i.accIn === 'tmem') {
    push({ kind: 'tcgen05.ld', label: 'tcgen05.ld → .reg', description: 'Move accumulator from TMEM into per-lane registers for epilogue (16dp×N shapes; lanes fan out across subpartitions).', ticks: 3 });
  }
  // Epilogue is really two sub-operations: stage accumulator fragments into
  // SMEM, then TMA-store that staging region to GMEM.
  push({
    kind: 'epilogue.stg_smem',
    label: 'acc → SMEM staging',
    description: 'Warp writes its accumulator fragments into an epilogue SMEM region (stmatrix / regular stores). Fills left-to-right.',
    ticks: 2,
  });
  push({
    kind: 'epilogue.tma.store',
    label: 'SMEM → GMEM (TMA store)',
    description:
      i.family === 'wgmma' || i.family === 'tcgen05' || i.family === 'tcgen05.block_scaled'
        ? 'cp.async.bulk.tensor.store drains the staging region to GMEM. Cells empty right-to-left.'
        : 'Plain st.global / stg: each thread writes its fragment to GMEM.',
    ticks: 2,
  });

  return base;
}

export const phases = computed<Phase[]>(() => phasesFor(inst.value, tileK.value));

export const totalTicks = computed<number>(() => {
  const p = phases.value;
  return p.length === 0 ? 1 : p[p.length - 1].endTick;
});

export const currentPhase = computed<Phase | null>(() => {
  const t = tick.value;
  return phases.value.find((p) => t >= p.startTick && t < p.endTick) ?? null;
});

// mma.step phases often appear in sequence k=0, k=1, ... — this computes the
// k-index of the current mma step (so the SMEM panel can highlight one column
// stripe at a time).
export const currentKStep = computed<{ k: number; total: number } | null>(() => {
  const ph = phases.value;
  const cur = currentPhase.value;
  if (!cur) return null;
  if (cur.kind !== 'wgmma.step' && cur.kind !== 'tcgen05.mma.step') return null;
  const steps = ph.filter((p) => p.kind === cur.kind);
  const k = steps.findIndex((p) => p.id === cur.id);
  return k >= 0 ? { k, total: steps.length } : null;
});
