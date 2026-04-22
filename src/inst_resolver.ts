// Resolve a human-friendly config (mode · shape · dtype · majors · flags)
// into an InstSpec from the catalog. Replaces the 200-item instruction
// dropdown with structured pickers.
//
// Major-ness semantics (cute):
//   K-major  → K is the innermost (fastest) axis in SMEM
//   MN-major → M (for A) / N (for B) is the innermost axis
// BLAS-style "A transposed" corresponds to different things for A and B;
// we avoid that code and expose A/B majors independently.

import {
  INSTRUCTIONS,
  type Dtype,
  type InstSpec,
  type Major,
  type OperandSource,
} from './instructions';

export type Mode = 'tcgen05' | 'wgmma' | 'mma' | 'wmma';

export interface ModeConfig {
  mode: Mode;
  M: number;
  N: number;
  K: number;
  dtypeA: Dtype;
  dtypeB: Dtype;
  accDtype: Dtype;
  majorA: Major;
  majorB: Major;
  // aSource is the SMEM / rmem / tmem choice for operand A. wgmma offers
  // SS (smem) vs RS (rmem); tcgen05 offers SS vs TS (tmem). CUTLASS
  // `mma_sm90_gmma.hpp:190-191` and `mma_sm100_umma.hpp:134,178` both
  // static_assert that the RS/TS variant requires a K-major A layout, so
  // callers should also force `majorA='K'` before resolving (see
  // `commitConfig` in state.ts).
  aSource?: OperandSource;
  ctaGroup?: 1 | 2;
  sparse?: boolean;
  warpSpecialized?: boolean;
}

export function modeOfInst(i: InstSpec): Mode {
  switch (i.family) {
    case 'wmma': return 'wmma';
    case 'mma': return 'mma';
    case 'wgmma': return 'wgmma';
    case 'tcgen05':
    case 'tcgen05.block_scaled': return 'tcgen05';
  }
}

// Find the InstSpec matching a structured config, or undefined.
export function resolveInst(c: ModeConfig): InstSpec | undefined {
  return INSTRUCTIONS.find(
    (i) =>
      modeOfInst(i) === c.mode &&
      i.M === c.M &&
      i.N === c.N &&
      i.K === c.K &&
      i.aDtypes.includes(c.dtypeA) &&
      i.bDtypes.includes(c.dtypeB) &&
      i.accDtypes.includes(c.accDtype) &&
      (i.ctaGroup ?? 1) === (c.ctaGroup ?? 1) &&
      !!i.sparse === !!c.sparse &&
      !!i.warpSpecialized === !!c.warpSpecialized &&
      i.majorA.includes(c.majorA) &&
      i.majorB.includes(c.majorB) &&
      (c.aSource === undefined || i.aSource.includes(c.aSource)),
  );
}

// Shapes available for a given mode (no filter on dtype / flags).
export function shapesForMode(mode: Mode): Array<{ M: number; N: number; K: number }> {
  const seen = new Set<string>();
  const out: Array<{ M: number; N: number; K: number }> = [];
  for (const i of INSTRUCTIONS) {
    if (modeOfInst(i) !== mode) continue;
    const key = `${i.M}x${i.N}x${i.K}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ M: i.M, N: i.N, K: i.K });
  }
  return out.sort((a, b) => a.M - b.M || a.N - b.N || a.K - b.K);
}

// Dtype options available for a given mode (all combined).
export function dtypesForMode(mode: Mode): { a: Dtype[]; b: Dtype[]; acc: Dtype[] } {
  const a = new Set<Dtype>(), b = new Set<Dtype>(), acc = new Set<Dtype>();
  for (const i of INSTRUCTIONS) {
    if (modeOfInst(i) !== mode) continue;
    i.aDtypes.forEach((d) => a.add(d));
    i.bDtypes.forEach((d) => b.add(d));
    i.accDtypes.forEach((d) => acc.add(d));
  }
  return { a: [...a], b: [...b], acc: [...acc] };
}

export function ctaGroupsForMode(mode: Mode): Array<1 | 2> {
  const s = new Set<1 | 2>();
  for (const i of INSTRUCTIONS) {
    if (modeOfInst(i) !== mode) continue;
    s.add((i.ctaGroup ?? 1) as 1 | 2);
  }
  return [...s].sort();
}

// Shapes restricted to what the current dtype + flags actually supports.
// `includePtxOnly` defaults to true so existing callers (e.g. tests) keep the
// full shape set; ConfigBar passes `false` to hide PTX-only wgmma shapes
// unless the user flips the "show all" toggle.
export function validShapesFor(
  mode: Mode,
  dtypeA: Dtype,
  dtypeB: Dtype,
  accDtype: Dtype,
  ctaGroup: 1 | 2,
  sparse: boolean,
  ws: boolean,
  includePtxOnly: boolean = true,
): Array<{ M: number; N: number; K: number; shapeSource: 'cutlass-atom' | 'ptx-only' }> {
  const seen = new Set<string>();
  const out: Array<{ M: number; N: number; K: number; shapeSource: 'cutlass-atom' | 'ptx-only' }> = [];
  for (const i of INSTRUCTIONS) {
    if (modeOfInst(i) !== mode) continue;
    if (!i.aDtypes.includes(dtypeA) || !i.bDtypes.includes(dtypeB) || !i.accDtypes.includes(accDtype)) continue;
    if ((i.ctaGroup ?? 1) !== ctaGroup) continue;
    if (!!i.sparse !== !!sparse) continue;
    if (!!i.warpSpecialized !== !!ws) continue;
    const shapeSource = i.shapeSource ?? 'cutlass-atom';
    if (!includePtxOnly && shapeSource === 'ptx-only') continue;
    const key = `${i.M}x${i.N}x${i.K}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ M: i.M, N: i.N, K: i.K, shapeSource });
  }
  return out.sort((a, b) => a.M - b.M || a.N - b.N || a.K - b.K);
}

export const DEFAULT_CONFIG_PER_MODE: Record<Mode, ModeConfig> = {
  tcgen05: {
    mode: 'tcgen05',
    M: 128, N: 128, K: 16,
    dtypeA: 'fp16', dtypeB: 'fp16', accDtype: 'fp32',
    majorA: 'K', majorB: 'K',
    ctaGroup: 1,
  },
  wgmma: {
    mode: 'wgmma',
    M: 64, N: 128, K: 16,
    dtypeA: 'fp16', dtypeB: 'fp16', accDtype: 'fp32',
    majorA: 'K', majorB: 'K',
  },
  mma: {
    mode: 'mma',
    M: 16, N: 8, K: 16,
    dtypeA: 'fp16', dtypeB: 'fp16', accDtype: 'fp32',
    majorA: 'K', majorB: 'MN',
  },
  wmma: {
    mode: 'wmma',
    M: 16, N: 16, K: 16,
    dtypeA: 'fp16', dtypeB: 'fp16', accDtype: 'fp32',
    majorA: 'K', majorB: 'K',
  },
};
