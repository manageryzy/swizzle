// TMEM model (sm_100). Geometry ground truth:
//   /tmp/cutlass/include/cute/pointer.hpp (tmem_ptr, 32-bit address layout)
//   /tmp/cutlass/include/cute/arch/tmem_allocator_sm100.hpp
//   /tmp/cutlass/include/cute/atom/copy_traits_sm100.hpp (TMEM_LOAD shapes)

export const TMEM_DP = 128; // DP (data-parallel) lanes per SM
export const TMEM_COLS = 512;
export const TMEM_SUBPARTITIONS = 4; // 128 DP / 32 lanes per subpartition

// A tcgen05.ld/.st shape: NdpNbit Mx → N DP lanes × bits-per-lane × M repeats.
export interface TcgShape {
  id: string;
  dp: number; // number of DP rows read per issue
  bits: number; // bits per DP row (per issue)
  repeat: number; // x1 / x2 / ... / x32
}

export const TCGEN05_SHAPES: TcgShape[] = [
  { id: '16dp128b x1', dp: 16, bits: 128, repeat: 1 },
  { id: '16dp128b x2', dp: 16, bits: 128, repeat: 2 },
  { id: '16dp128b x4', dp: 16, bits: 128, repeat: 4 },
  { id: '16dp128b x8', dp: 16, bits: 128, repeat: 8 },
  { id: '16dp256b x1', dp: 16, bits: 256, repeat: 1 },
  { id: '16dp256b x2', dp: 16, bits: 256, repeat: 2 },
  { id: '16dp256b x4', dp: 16, bits: 256, repeat: 4 },
  { id: '16dp256b x8', dp: 16, bits: 256, repeat: 8 },
  { id: '16dp256b x16', dp: 16, bits: 256, repeat: 16 },
  { id: '16dp256b x32', dp: 16, bits: 256, repeat: 32 },
];

export function findShape(id: string): TcgShape | undefined {
  return TCGEN05_SHAPES.find((s) => s.id === id);
}

// Accumulator footprint in TMEM for a given (M, N, cta_group).
// Per PTX, each CTA in the cta_group holds (M / cta_group) DP × N cols.
export function accFootprint(M: number, N: number, ctaGroup: 1 | 2 = 1): {
  dp: number;
  cols: number;
} {
  return { dp: Math.min(TMEM_DP, M / ctaGroup), cols: Math.min(TMEM_COLS, N) };
}

// Subpartition of a DP row: 4 subpartitions of 32 DP each.
export function subpartitionOf(dp: number): number {
  return Math.floor(dp / 32);
}
