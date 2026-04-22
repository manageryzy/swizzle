// Matrix descriptors for wgmma (sm_90) and tcgen05.mma (sm_100).
// They are structurally similar but have DIFFERENT bitfields and DIFFERENT
// layout_type enum values. See:
//   /tmp/cutlass/include/cute/arch/mma_sm90_desc.hpp  (GmmaDescriptor)
//   /tmp/cutlass/include/cute/arch/mma_sm100_desc.hpp (SmemDescriptor)

import { Swizzle } from './swizzle';

export type DescArch = 'sm90' | 'sm100';

export interface DescInput {
  arch: DescArch;
  startByte: number;
  leadingByteOffset: number; // in 16B units
  strideByteOffset: number; // in 16B units
  swizzle: Swizzle;
  baseOffset?: number;
  lboMode?: 0 | 1; // sm_100 only
  version?: number; // sm_100 only
}

export interface DescField {
  name: string;
  bits: [number, number];
  value: number;
}

export interface BuiltDesc {
  arch: DescArch;
  raw: bigint;
  fields: DescField[];
  layoutTypeName: string;
}

// sm_90 GmmaDescriptor LayoutType (2 bits):
//   SWIZZLE_NONE=0, SWIZZLE_128B=1, SWIZZLE_64B=2, SWIZZLE_32B=3
// Accepts either byte-level (M=1, S=3) or bit-level (M=4, S=3) Swizzle inputs.
function sm90LayoutType(sw: Swizzle): { value: number; name: string } {
  const { B, M, S } = sw;
  if ((M === 1 || M === 4) && S === 3) {
    if (B === 0) return { value: 0, name: 'SWIZZLE_NONE' };
    if (B === 3) return { value: 1, name: 'SWIZZLE_128B' };
    if (B === 2) return { value: 2, name: 'SWIZZLE_64B' };
    if (B === 1) return { value: 3, name: 'SWIZZLE_32B' };
  }
  throw new Error(`sm_90 has no layout_type for Swizzle<${B},${M},${S}> (128B_BASE32B is sm_100+)`);
}

// sm_100 SmemDescriptor LayoutType (3 bits):
//   SWIZZLE_NONE=0, SWIZZLE_128B_BASE32B=1, SWIZZLE_128B=2, SWIZZLE_64B=4, SWIZZLE_32B=6
function sm100LayoutType(sw: Swizzle): { value: number; name: string } {
  const { B, M, S } = sw;
  if ((M === 1 || M === 4) && S === 3) {
    if (B === 0) return { value: 0, name: 'SWIZZLE_NONE' };
    if (B === 1) return { value: 6, name: 'SWIZZLE_32B' };
    if (B === 2) return { value: 4, name: 'SWIZZLE_64B' };
    if (B === 3) return { value: 2, name: 'SWIZZLE_128B' };
  }
  if ((M === 2 || M === 5) && S === 2 && B === 2) return { value: 1, name: 'SWIZZLE_128B_BASE32B' };
  throw new Error(`no sm_100 layout_type for Swizzle<${B},${M},${S}>`);
}

function assemble(fields: DescField[]): bigint {
  let raw = 0n;
  for (const f of fields) raw |= BigInt(f.value) << BigInt(f.bits[0]);
  return raw;
}

export function buildDescriptor(input: DescInput): BuiltDesc {
  if (input.arch === 'sm90') {
    const lt = sm90LayoutType(input.swizzle);
    const fields: DescField[] = [
      { name: 'start_address', bits: [0, 14], value: (input.startByte >>> 4) & 0x3fff },
      { name: 'leading_byte_offset', bits: [16, 30], value: input.leadingByteOffset & 0x3fff },
      { name: 'stride_byte_offset', bits: [32, 46], value: input.strideByteOffset & 0x3fff },
      { name: 'base_offset', bits: [49, 52], value: (input.baseOffset ?? 0) & 0x7 },
      { name: 'layout_type', bits: [62, 64], value: lt.value & 0x3 },
    ];
    return { arch: 'sm90', raw: assemble(fields), fields, layoutTypeName: lt.name };
  } else {
    const lt = sm100LayoutType(input.swizzle);
    // PTX Table 42 reserves three zero-bands in the 64-bit descriptor. We
    // surface them as explicit zero-valued DescField rows so the bitfield
    // panel shows the full bit budget rather than mysterious gaps — this
    // also forces `assemble()` to annotate those bits if anyone ever
    // forgets to keep them clear.
    const fields: DescField[] = [
      { name: 'start_address', bits: [0, 14], value: (input.startByte >>> 4) & 0x3fff },
      { name: 'reserved_0', bits: [14, 16], value: 0 },
      { name: 'leading_byte_offset', bits: [16, 30], value: input.leadingByteOffset & 0x3fff },
      { name: 'reserved_1', bits: [30, 32], value: 0 },
      { name: 'stride_byte_offset', bits: [32, 46], value: input.strideByteOffset & 0x3fff },
      { name: 'version', bits: [46, 48], value: (input.version ?? 1) & 0x3 },
      { name: 'base_offset', bits: [49, 52], value: (input.baseOffset ?? 0) & 0x7 },
      { name: 'lbo_mode', bits: [52, 53], value: (input.lboMode ?? 0) & 0x1 },
      { name: 'reserved_2', bits: [53, 60], value: 0 },
      { name: 'layout_type', bits: [61, 64], value: lt.value & 0x7 },
    ];
    return { arch: 'sm100', raw: assemble(fields), fields, layoutTypeName: lt.name };
  }
}
