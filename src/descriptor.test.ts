import { describe, expect, it } from 'vitest';
import { buildDescriptor } from './descriptor';
import { NO_SWIZZLE, SW128, SW64, SW32 } from './swizzle';

describe('sm_90 GmmaDescriptor — 2-bit layout_type', () => {
  it('layout_type encoding matches mma_sm90_desc.hpp comment', () => {
    // sm_90 enum: NONE=0, 128B=1, 64B=2, 32B=3
    expect(buildDescriptor({ arch: 'sm90', startByte: 0, leadingByteOffset: 0, strideByteOffset: 0, swizzle: NO_SWIZZLE }).layoutTypeName).toBe('SWIZZLE_NONE');
    expect(buildDescriptor({ arch: 'sm90', startByte: 0, leadingByteOffset: 0, strideByteOffset: 0, swizzle: SW128 }).layoutTypeName).toBe('SWIZZLE_128B');
    expect(buildDescriptor({ arch: 'sm90', startByte: 0, leadingByteOffset: 0, strideByteOffset: 0, swizzle: SW64 }).layoutTypeName).toBe('SWIZZLE_64B');
    expect(buildDescriptor({ arch: 'sm90', startByte: 0, leadingByteOffset: 0, strideByteOffset: 0, swizzle: SW32 }).layoutTypeName).toBe('SWIZZLE_32B');
  });

  it('golden hex: SW128, addr=0x100, LBO=16, SBO=64', () => {
    // start_address = 0x100 >> 4 = 0x10 at bits [0,14)
    // leading_byte_offset = 0x10 at bits [16,30)
    // stride_byte_offset  = 0x40 at bits [32,46)
    // layout_type = 1 (SWIZZLE_128B) at bits [62,64)
    const b = buildDescriptor({
      arch: 'sm90',
      startByte: 0x100,
      leadingByteOffset: 0x10,
      strideByteOffset: 0x40,
      swizzle: SW128,
    });
    const expected =
      (0x10n << 0n) | (0x10n << 16n) | (0x40n << 32n) | (1n << 62n);
    expect(b.raw).toBe(expected);
  });
});

describe('sm_100 SmemDescriptor — 3-bit layout_type', () => {
  it('layout_type encoding matches mma_sm100_desc.hpp', () => {
    // sm_100 enum: NONE=0, 128B_BASE32B=1, 128B=2, 64B=4, 32B=6
    expect(buildDescriptor({ arch: 'sm100', startByte: 0, leadingByteOffset: 0, strideByteOffset: 0, swizzle: NO_SWIZZLE }).layoutTypeName).toBe('SWIZZLE_NONE');
    expect(buildDescriptor({ arch: 'sm100', startByte: 0, leadingByteOffset: 0, strideByteOffset: 0, swizzle: SW128 }).layoutTypeName).toBe('SWIZZLE_128B');
    expect(buildDescriptor({ arch: 'sm100', startByte: 0, leadingByteOffset: 0, strideByteOffset: 0, swizzle: SW64 }).layoutTypeName).toBe('SWIZZLE_64B');
    expect(buildDescriptor({ arch: 'sm100', startByte: 0, leadingByteOffset: 0, strideByteOffset: 0, swizzle: SW32 }).layoutTypeName).toBe('SWIZZLE_32B');
  });

  it('golden hex: SW128, addr=0x100, LBO=16, SBO=64, version=1', () => {
    // bit positions (mma_sm100_desc.hpp):
    //   start_address      [0, 14)
    //   leading_byte_offset[16,30)
    //   stride_byte_offset [32,46)
    //   version            [46,48)
    //   layout_type        [61,64), value 2 for SWIZZLE_128B
    const b = buildDescriptor({
      arch: 'sm100',
      startByte: 0x100,
      leadingByteOffset: 0x10,
      strideByteOffset: 0x40,
      swizzle: SW128,
    });
    const expected =
      (0x10n << 0n) |
      (0x10n << 16n) |
      (0x40n << 32n) |
      (1n << 46n) | // version
      (2n << 61n); // layout_type = SWIZZLE_128B = 2
    expect(b.raw).toBe(expected);
  });

  it('sm_100 exposes the new 128B_BASE32B layout; sm_90 does not', () => {
    expect(() => buildDescriptor({
      arch: 'sm100', startByte: 0, leadingByteOffset: 0, strideByteOffset: 0,
      swizzle: { B: 2, M: 5, S: 2 },
    })).not.toThrow();
    expect(() => buildDescriptor({
      arch: 'sm90', startByte: 0, leadingByteOffset: 0, strideByteOffset: 0,
      swizzle: { B: 2, M: 5, S: 2 },
    })).toThrow();
  });
});
