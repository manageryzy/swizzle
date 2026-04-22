import { describe, expect, it } from 'vitest';
import {
  NO_SWIZZLE,
  SW128,
  SW128_BASE32B,
  SW32,
  SW64,
  LayoutType,
  apply,
  bankOfByte,
  layoutTypeOf,
} from './swizzle';

describe('cute::Swizzle<B,M,S>::apply', () => {
  it('NO_SWIZZLE (B=0) is identity', () => {
    for (const off of [0, 4, 0x80, 0x100, 0x180, 0xabcd]) {
      expect(apply(NO_SWIZZLE, off)).toBe(off);
    }
  });

  it('SW128 = Swizzle<3,1,3> at byte level: yyy = bits [4..6], zzz = bits [1..3]', () => {
    // yyy_msk = 0b111 << 4 = 0x70; result = o ^ ((o & 0x70) >> 3)
    // Span covers 128 B within a line, which is the whole point of the name.
    const cases: [number, number][] = [
      [0x00, 0x00],
      [0x10, 0x10 ^ 0x02], // 0x12
      [0x20, 0x20 ^ 0x04], // 0x24
      [0x30, 0x30 ^ 0x06], // 0x36
      [0x40, 0x40 ^ 0x08], // 0x48
      [0x70, 0x70 ^ 0x0e], // 0x7e
      [0x80, 0x80],        // above 128 B: bit 4-6 empty, no XOR
    ];
    for (const [off, want] of cases) expect(apply(SW128, off)).toBe(want);
  });

  it('SW64 = Swizzle<2,1,3>: 2-bit XOR within 64 B span', () => {
    // yyy_msk = 3 << 4 = 0x30
    expect(apply(SW64, 0x10)).toBe(0x10 ^ 0x02);
    expect(apply(SW64, 0x20)).toBe(0x20 ^ 0x04);
    expect(apply(SW64, 0x30)).toBe(0x30 ^ 0x06);
    expect(apply(SW64, 0x40)).toBe(0x40); // outside 64B span
  });

  it('SW32 = Swizzle<1,1,3>: 1-bit XOR within 32 B span', () => {
    // yyy_msk = 1 << 4 = 0x10
    expect(apply(SW32, 0x10)).toBe(0x10 ^ 0x02);
    expect(apply(SW32, 0x20)).toBe(0x20); // bit 4 = 0
    expect(apply(SW32, 0x30)).toBe(0x30 ^ 0x02);
  });

  it('swizzle is an involution', () => {
    for (const sw of [SW32, SW64, SW128, SW128_BASE32B]) {
      for (const off of [0, 0x4, 0x10, 0x80, 0x100, 0x1abc, 0xdead]) {
        expect(apply(sw, apply(sw, off))).toBe(off);
      }
    }
  });

  it('SW128 preserves the low M=1 bit (2-byte fp16 alignment)', () => {
    // Byte-level M=1 means bit 0 is untouched — i.e. 2-byte element boundary.
    for (const lane of [0, 1]) {
      expect(apply(SW128, 0x10 | lane) & 0x1).toBe(lane);
    }
  });
});

describe('layoutTypeOf (SmemDescriptor 3-bit field)', () => {
  it('matches the mma_sm100_desc.hpp enum', () => {
    expect(layoutTypeOf(NO_SWIZZLE)).toBe(LayoutType.SWIZZLE_NONE);
    expect(layoutTypeOf(SW32)).toBe(LayoutType.SWIZZLE_32B);
    expect(layoutTypeOf(SW64)).toBe(LayoutType.SWIZZLE_64B);
    expect(layoutTypeOf(SW128)).toBe(LayoutType.SWIZZLE_128B);
    expect(layoutTypeOf(SW128_BASE32B)).toBe(LayoutType.SWIZZLE_128B_BASE32B);
  });
});

describe('bankOfByte (32 banks × 4B)', () => {
  it('wraps modulo 32 over 4-byte words', () => {
    expect(bankOfByte(0)).toBe(0);
    expect(bankOfByte(4)).toBe(1);
    expect(bankOfByte(124)).toBe(31);
    expect(bankOfByte(128)).toBe(0); // next 128B line
  });

  it('SW128 permutes 16-byte slots within a 128B line', () => {
    // Within one 128B line there are 8 × 16B slots. SW128 maps each to a
    // distinct bank-word group via the yyy→zzz XOR.
    const slots = [0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70];
    const naiveBanks = new Set<number>(slots.map(bankOfByte));
    const swBanks = new Set<number>(slots.map((o) => bankOfByte(apply(SW128, o))));
    // Naive step-16 already hits 8 distinct banks (bytes 0,16,32... → banks
    // 0,4,8,12,16,20,24,28). SW128 redistributes them to 8 *different* banks.
    expect(naiveBanks.size).toBe(8);
    expect(swBanks.size).toBe(8);
    // The two sets are actually identical (permutation), but the *mapping*
    // from lane → bank differs, which is what resolves cross-line conflicts.
  });
});
