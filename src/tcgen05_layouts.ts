// PTX tcgen05 TMEM accumulator layouts A–G.
// Source: PTX ISA §9.7.16.10.5 — "Valid Combinations of M × cta_group × sparsity".
// Each layout describes how the M×N accumulator maps into the 128 DP × 512 col
// TMEM grid of one SM.

export type PtxLayout = 'A' | 'B' | 'C' | 'D' | 'E' | 'F' | 'G';

export interface LayoutInfo {
  id: PtxLayout;
  name: string;
  conditions: string;
  dpRows: number;
  description: string;
}

export const PTX_LAYOUTS: Record<PtxLayout, LayoutInfo> = {
  A: {
    id: 'A',
    name: 'Layout A',
    conditions: 'M=256, cta_group::2, dense A',
    dpRows: 128, // M=256 split across 2 CTAs = 128 DP each
    description:
      'M=256 dense MMA, 2-CTA pair. Each CTA holds 128 DP rows of accumulator; the pair covers all 256 output rows.',
  },
  B: {
    id: 'B',
    name: 'Layout B',
    conditions: 'M=128, cta_group::2, dense A',
    dpRows: 64,
    description:
      'M=128 dense MMA, 2-CTA. Each CTA holds 64 DP rows; one subpartition pair is unused.',
  },
  C: {
    id: 'C',
    name: 'Layout C',
    conditions: 'M=128, cta_group::2, sparse A',
    dpRows: 128,
    description:
      'M=128 sparse MMA, 2-CTA. Sparse metadata doubles the logical M, so both CTAs use their full 128 DP.',
  },
  D: {
    id: 'D',
    name: 'Layout D',
    conditions: 'M=128, cta_group::1',
    dpRows: 128,
    description:
      'M=128 single-CTA MMA. All 128 DP rows of one SM hold the accumulator.',
  },
  E: {
    id: 'E',
    name: 'Layout E',
    conditions: 'M=64, cta_group::1, .ws (warp-specialized)',
    dpRows: 64,
    description:
      'M=64 warp-specialized. Bottom 64 DP rows of the subpartition hold the accumulator.',
  },
  F: {
    id: 'F',
    name: 'Layout F',
    conditions: 'M=64, cta_group::1, non-ws',
    dpRows: 64,
    description:
      'M=64 standard single-CTA. Top 64 DP rows used; different sub-partition assignment than Layout E.',
  },
  G: {
    id: 'G',
    name: 'Layout G',
    conditions: 'M=32, cta_group::1, .ws (warp-specialized)',
    dpRows: 32,
    description:
      'M=32 narrow MMA, warp-specialized only. Per PTX Table 41, M=32 is reachable only via .ws; uses a single subpartition (32 DP) of TMEM per CTA.',
  },
};

export function classifyLayout(
  M: number,
  ctaGroup: 1 | 2,
  opts: { sparse?: boolean; warpSpecialized?: boolean } = {},
): PtxLayout | null {
  const { sparse = false, warpSpecialized = false } = opts;
  if (M === 256 && ctaGroup === 2 && !sparse) return 'A';
  if (M === 128 && ctaGroup === 2 && !sparse) return 'B';
  if (M === 128 && ctaGroup === 2 && sparse) return 'C';
  if (M === 128 && ctaGroup === 1) return 'D';
  if (M === 64 && ctaGroup === 1 && warpSpecialized) return 'E';
  if (M === 64 && ctaGroup === 1 && !warpSpecialized) return 'F';
  if (M === 32 && ctaGroup === 1 && warpSpecialized) return 'G';
  return null;
}

// The M-range of the accumulator that a given peer CTA holds in its TMEM.
// Returns [mLo, mHi) for `peer ∈ {0, 1}` when cta_group::2; peer 0 only for
// cta_group::1. Based on PTX §9.7.16.10.5 semantics.
export function peerMRange(
  layout: PtxLayout,
  peer: 0 | 1,
  M: number,
): { lo: number; hi: number; peerLabel: string } {
  switch (layout) {
    case 'A': // M=256, cg2: split 128/128
      return peer === 0
        ? { lo: 0, hi: 128, peerLabel: 'CTA 0 of 2' }
        : { lo: 128, hi: 256, peerLabel: 'CTA 1 of 2' };
    case 'B': // M=128, cg2 dense: split 64/64
    case 'C': // M=128, cg2 sparse: (logical M is 2× physical)
      return peer === 0
        ? { lo: 0, hi: M / 2, peerLabel: 'CTA 0 of 2' }
        : { lo: M / 2, hi: M, peerLabel: 'CTA 1 of 2' };
    case 'D':
    case 'E':
    case 'F':
    case 'G':
    default:
      return { lo: 0, hi: M, peerLabel: 'single CTA' };
  }
}
