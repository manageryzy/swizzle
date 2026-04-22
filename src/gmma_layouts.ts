// Canonical SMEM layouts for the GMMA/UMMA descriptor builders.
// Verbatim from:
//   /tmp/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp (sm_90 WGMMA)
//   /tmp/cutlass/include/cute/atom/mma_traits_sm100.hpp (sm_100 UMMA)
//
// T is the element count per 128-bit unit = 16 / sizeof(elem).
// SBO = stride byte offset, LBO = leading byte offset (16 B units).

import type { Major } from './instructions';
import type { SwizzleKind } from './swizzle';

export type DescArch = 'sm90' | 'sm100';

export interface CanonicalLayout {
  swizzle: string;
  shape: string;
  stride: string;
  comment: string;
}

const SM90_MN: Record<SwizzleKind, CanonicalLayout | null> = {
  none: {
    swizzle: 'Swizzle<0,4,3>',
    shape: '((1,n),(8,k))',
    stride: '((T,SBO),(1,LBO))',
    comment: 'INTERLEAVE — 1×(8,k) brick per uint128. SBO=stride, LBO=leading.',
  },
  '32B': {
    swizzle: 'Swizzle<1,4,3>',
    shape: '((2,n),(8,k))',
    stride: '((1,LBO),(2,SBO))',
    comment: 'B32 — 32-byte swizzle span; brick is 2×(8,k) per uint128.',
  },
  '64B': {
    swizzle: 'Swizzle<2,4,3>',
    shape: '((4,n),(8,k))',
    stride: '((1,LBO),(4,SBO))',
    comment: 'B64 — 64-byte span; brick is 4×(8,k).',
  },
  '128B': {
    swizzle: 'Swizzle<3,4,3>',
    shape: '((8,n),(8,k))',
    stride: '((1,LBO),(8,SBO))',
    comment: 'B128 — 128-byte span; brick is 8×(8,k), canonical for fp16 wgmma.',
  },
  '128B.base32B': null,
};

const SM90_K: Record<SwizzleKind, CanonicalLayout | null> = {
  none: {
    swizzle: 'Swizzle<0,4,3>',
    shape: '((8,n),2)',
    stride: '((1,SBO),LBO)',
    comment: 'INTERLEAVE — 2 uint128 per K tile.',
  },
  '32B': {
    swizzle: 'Swizzle<1,4,3>',
    shape: '((8,n),2)',
    stride: '((2,SBO),(1,T))',
    comment: 'B32 K-major — 32B swizzle aligns 2×T-element rows.',
  },
  '64B': {
    swizzle: 'Swizzle<2,4,3>',
    shape: '((8,n),2)',
    stride: '((4,SBO),(1,T))',
    comment: 'B64 K-major.',
  },
  '128B': {
    swizzle: 'Swizzle<3,4,3>',
    shape: '((8,n),2)',
    stride: '((8,SBO),(1,T))',
    comment: 'B128 K-major — canonical fp16 wgmma choice.',
  },
  '128B.base32B': null,
};

const SM100_MN: Record<SwizzleKind, CanonicalLayout | null> = {
  none: SM90_MN.none,
  '32B': SM90_MN['32B'],
  '64B': SM90_MN['64B'],
  '128B': SM90_MN['128B'],
  '128B.base32B': {
    swizzle: 'Swizzle<2,5,2>',
    shape: '((T,8,m),(4,k))',
    stride: '((1,T,LBO),(?,SBO))',
    comment: '128B_BASE32B — sm_100-only variant with 32-byte alignment base.',
  },
};

// PTX Table 42 says layout_type=1 (128B_BASE32B) is valid for both K-major
// and MN-major descriptors, but CUTLASS `mma_traits_sm100.hpp` only ships an
// MN-major canonical atom. Mark the K-major slot as a documented gap rather
// than silently returning `null` (which the panel renders as "n/a" without
// context). Callers that encounter this should fall back to `SM100_K['128B']`.
const SM100_K_128B_BASE32B_NOTE: CanonicalLayout = {
  swizzle: 'Swizzle<2,5,2>',
  shape: 'n/a',
  stride: 'n/a',
  comment:
    '128B_BASE32B K-major — valid per PTX Table 42 (layout_type=1), but no canonical cute atom ships for this combination. Use B128 K-major when possible.',
};

const SM100_K: Record<SwizzleKind, CanonicalLayout | null> = {
  ...SM90_K,
  '128B.base32B': SM100_K_128B_BASE32B_NOTE,
};

export function canonicalLayout(
  arch: DescArch,
  major: Major,
  swizzle: SwizzleKind,
): CanonicalLayout | null {
  if (arch === 'sm90') return major === 'MN' ? SM90_MN[swizzle] : SM90_K[swizzle];
  return major === 'MN' ? SM100_MN[swizzle] : SM100_K[swizzle];
}

// Canonical "T" per element dtype — how many elements fit in a uint128.
export function canonicalT(bytesPerElement: number): number {
  return Math.max(1, Math.floor(16 / Math.max(1, bytesPerElement)));
}
