// Port of cute::Swizzle<B,M,S> — see
//   /tmp/cutlass/include/cute/swizzle.hpp
//   /tmp/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp (canonical atoms)
//   /tmp/cutlass/include/cute/arch/mma_sm100_desc.hpp (LayoutType enum)
//
// Semantics (verbatim from swizzle.hpp):
//   apply(offset) = offset XOR shiftr(offset & yyy_msk, S)
//   yyy_msk = ((1 << B) - 1) << (M + max(0, S))
// B = number of swizzle bits, M = base offset, S = shift between Y and Z bit groups.

export interface Swizzle {
  readonly B: number;
  readonly M: number;
  readonly S: number;
}

export function apply(sw: Swizzle, offset: number): number {
  const { B, M, S } = sw;
  if (B === 0) return offset;
  const bitMask = (1 << B) - 1;
  const yyyShift = M + Math.max(0, S);
  const yyyMask = bitMask << yyyShift;
  const picked = (offset & yyyMask) >>> S;
  return (offset ^ picked) >>> 0;
}

// Canonical atoms (from mma_traits_sm90_gmma.hpp + mma_traits_sm100.hpp),
// given here in BYTE-OFFSET form — upcast<8>(bit-level atom).
//
// Cute defines atoms at bit granularity, e.g. `Layout_K_SW128_Atom_Bits` uses
// `Swizzle<3,4,3>` over bit offsets. Our demo operates on byte offsets (since
// SMEM banks are 4 B words), so we apply `upcast<8>` which drops M by 3:
//   bit-level  Swizzle<B, M_bit, S>
//   byte-level Swizzle<B, M_bit - 3, S>
//
// As a result: M_byte = 1 preserves fp16 (2 B) element alignment — exactly
// what cute's bit-level M=4 means (preserve the low 16 bits = 2 B). The
// swizzle span in bytes is 2^(M_byte + S + B) = 128 B for SW128, etc.
export const NO_SWIZZLE: Swizzle = { B: 0, M: 1, S: 3 }; // INTER — identity
export const SW32: Swizzle = { B: 1, M: 1, S: 3 };       // 32B span
export const SW64: Swizzle = { B: 2, M: 1, S: 3 };       // 64B span
export const SW128: Swizzle = { B: 3, M: 1, S: 3 };      // 128B span
export const SW128_BASE32B: Swizzle = { B: 2, M: 2, S: 2 }; // sm_100-only 32B-base

export type SwizzleKind =
  | 'none'
  | '32B'
  | '64B'
  | '128B'
  | '128B.base32B';

// Canonical atoms above are defined for fp16 (2-byte elements, M=1 at byte
// level). For other element sizes the preserved region M shifts so that the
// element boundary stays aligned — this matches cute's
// `Layout_K_SW128_Atom<T> = upcast<sizeof_bits<T>>(...)` behaviour.
//
//   M_byte = log2(elemBytes)
//
// fp8  (1B) → M_byte = 0
// fp16 (2B) → M_byte = 1
// tf32/fp32 (4B) → M_byte = 2
// fp64 (8B) → M_byte = 3
// For sub-byte dtypes (fp4/fp6/int4) cute packs multiple elements into a
// byte; we clamp M_byte to 0 which preserves byte alignment.
export function effectiveSwizzle(kind: SwizzleKind, elemBytes: number): Swizzle {
  const eb = Math.max(1, elemBytes);
  const M = Math.max(0, Math.round(Math.log2(eb)));
  switch (kind) {
    case 'none': return { B: 0, M, S: 3 };
    case '32B': return { B: 1, M, S: 3 };
    case '64B': return { B: 2, M, S: 3 };
    case '128B': return { B: 3, M, S: 3 };
    // 128B_BASE32B has a wider preserved base (M+1 over the fp16 canonical).
    case '128B.base32B': return { B: 2, M: M + 1, S: 2 };
  }
}

export const SWIZZLES: Record<SwizzleKind, Swizzle> = {
  none: NO_SWIZZLE,
  '32B': SW32,
  '64B': SW64,
  '128B': SW128,
  '128B.base32B': SW128_BASE32B,
};

// LayoutType field of the 64-bit SmemDescriptor (3 bits, at [61,64)).
//   See mma_sm100_desc.hpp lines 79–85.
export enum LayoutType {
  SWIZZLE_NONE = 0,
  SWIZZLE_128B_BASE32B = 1,
  SWIZZLE_128B = 2,
  SWIZZLE_64B = 4,
  SWIZZLE_32B = 6,
}

export function layoutTypeOf(sw: Swizzle): LayoutType {
  const { B, M, S } = sw;
  // Byte-level (current impl): M=1 for the 128B/64B/32B family.
  if (M === 1 && S === 3) {
    if (B === 0) return LayoutType.SWIZZLE_NONE;
    if (B === 1) return LayoutType.SWIZZLE_32B;
    if (B === 2) return LayoutType.SWIZZLE_64B;
    if (B === 3) return LayoutType.SWIZZLE_128B;
  }
  if (M === 2 && S === 2 && B === 2) return LayoutType.SWIZZLE_128B_BASE32B;
  // Bit-level (legacy) — accept too so tests that pass bit-level values pass.
  if (M === 4 && S === 3) {
    if (B === 0) return LayoutType.SWIZZLE_NONE;
    if (B === 1) return LayoutType.SWIZZLE_32B;
    if (B === 2) return LayoutType.SWIZZLE_64B;
    if (B === 3) return LayoutType.SWIZZLE_128B;
  }
  if (M === 5 && S === 2 && B === 2) return LayoutType.SWIZZLE_128B_BASE32B;
  throw new Error(`Swizzle<${B},${M},${S}> has no corresponding LayoutType`);
}

// Shared memory: 32 banks × 4-byte words. Byte offset → bank id.
export function bankOfByte(byteOffset: number): number {
  return (byteOffset >>> 2) & 31;
}

// Convenience: apply a swizzle in byte space to a (row, col) index for a tile
// with given row stride (in bytes) and return the bank.
export function bankForCell(
  sw: Swizzle,
  row: number,
  col: number,
  rowStrideBytes: number,
  elemBytes: number,
): number {
  const logical = row * rowStrideBytes + col * elemBytes;
  return bankOfByte(apply(sw, logical));
}
