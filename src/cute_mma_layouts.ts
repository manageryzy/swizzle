// Real cute CLayouts ported verbatim from CUTLASS source.
//
// Ground truth:
//   /tmp/cutlass/include/cute/atom/mma_traits_sm80.hpp (SM80_16x8_Row)
//   /tmp/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp (CLayout_64xN)
//
// A CLayout maps (threadId, valueId) → element index in the output tile,
// where the index is interpreted as COLUMN-MAJOR in an MxN matrix (m = off%M,
// n = off/M). This matches how cute partitions the accumulator.

import { IntTuple, layoutAt, sizeOf } from './cute_layout';

export interface CLayout {
  readonly label: string;
  readonly sourceFile: string;
  readonly shape: IntTuple; // [threadDim, valueDim]
  readonly stride: IntTuple;
  readonly M: number;
  readonly N: number;
  readonly threads: number;
  readonly valuesPerThread: number;
}

// SM80 m16n8k* — all use SM80_16x8_Row:
//   Layout<Shape<Shape<_4,_8>,Shape<_2,_2>>, Stride<Stride<_32,_1>,Stride<_16,_8>>>
const SM80_16x8_Row: CLayout = {
  label: 'SM80_16x8_Row',
  sourceFile: 'cute/atom/mma_traits_sm80.hpp',
  shape: [[4, 8], [2, 2]],
  stride: [[32, 1], [16, 8]],
  M: 16,
  N: 8,
  threads: 32,
  valuesPerThread: 4,
};

// SM90 wgmma m64nNk16 / k32:
//   CLayout_64xN = Layout<Shape<Shape<_4,_8,_4>,Shape<_2,_2,Int<N/8>>>,
//                         Stride<Stride<_128,_1,_16>,Stride<_64,_8,_512>>>
function sm90CLayout64xN(N: number): CLayout {
  return {
    label: `CLayout_64x${N}`,
    sourceFile: 'cute/atom/mma_traits_sm90_gmma.hpp',
    shape: [[4, 8, 4], [2, 2, N / 8]],
    stride: [[128, 1, 16], [64, 8, 512]],
    M: 64,
    N,
    threads: 128,
    valuesPerThread: 4 * (N / 8),
  };
}

// Registry keyed by instruction id from ./instructions.ts. Only instructions
// with a real CLayout port have an entry; others fall back to the approximate
// owner-map in RmemPanel.
export const C_LAYOUTS: Record<string, CLayout> = {
  // sm_80 mma (all share SM80_16x8_Row)
  'sm80.mma.m16n8k16.f16': SM80_16x8_Row,
  'sm80.mma.m16n8k8.tf32': SM80_16x8_Row,
  'sm89.mma.m16n8k32.fp8': SM80_16x8_Row,

  // sm_90 wgmma — one entry per (N, acc dtype). CUTLASS ships the 8-atom
  // subset; F7 extends the catalog to the full PTX set (every multiple of 8
  // from 8 to 256). Both atom and PTX-only variants project through the same
  // CLayout_64xN so we reuse `sm90CLayout64xN` across the board.
  ...Object.fromEntries(
    Array.from({ length: 32 }, (_, i) => (i + 1) * 8).flatMap((N) => [
      [`sm90.wgmma.m64n${N}k16.f32f16`, sm90CLayout64xN(N)],
      [`sm90.wgmma.m64n${N}k16.f16f16`, sm90CLayout64xN(N)],
    ]),
  ),
};

export function clayoutOf(instId: string): CLayout | undefined {
  return C_LAYOUTS[instId];
}

// For each (m, n) in the MxN output, compute the owning thread. Returns
// null for instructions without a ported CLayout (RmemPanel uses its fallback
// in that case).
export function ownershipMap(c: CLayout): (m: number, n: number) => number {
  // Precompute offset → thread table.
  const total = sizeOf(c.shape);
  const threadOf = new Int16Array(c.M * c.N).fill(-1);
  for (let tid = 0; tid < c.threads; tid++) {
    for (let vid = 0; vid < c.valuesPerThread; vid++) {
      // idx = tid + vid * threads (since shape = [[thread], [value]],
      // thread mode comes first and cycles fastest in column-major).
      const idx = tid + vid * c.threads;
      if (idx >= total) break;
      const off = layoutAt(c.shape, c.stride, idx);
      const m = off % c.M;
      const n = Math.floor(off / c.M);
      if (m < c.M && n < c.N) threadOf[n * c.M + m] = tid;
    }
  }
  return (m, n) => {
    const t = threadOf[n * c.M + m];
    return t < 0 ? 0 : t;
  };
}
