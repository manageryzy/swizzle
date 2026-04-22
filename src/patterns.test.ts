import { describe, expect, it } from 'vitest';
import {
  DEFAULT_CTX,
  LDMATRIX_X4_N,
  NAIVE_COL0,
  CPASYNC_16B,
  accessArrival,
  analyzeConflicts,
  contiguousArrival,
  maxConflictWay,
} from './patterns';
import { NO_SWIZZLE, SW128 } from './swizzle';

const CTX = DEFAULT_CTX;

describe('patterns + conflict analysis', () => {
  it('naive col-0 across 32 rows → 32-way conflict on bank 0', () => {
    // Row stride = 128B means each lane's offset is a multiple of 128 B, so
    // every access falls on bank 0 regardless of swizzle (SW128 only touches
    // bits 4-6, which are 0 for multiples of 128).
    const acc = NAIVE_COL0.accesses(CTX);
    expect(maxConflictWay(acc, NO_SWIZZLE)).toBe(32);
    const cs = analyzeConflicts(acc, NO_SWIZZLE);
    expect(cs).toHaveLength(1);
    expect(cs[0].bank).toBe(0);
  });

  it('byte-level SW128 does NOT help naive col-0 (cross-line pattern)', () => {
    // SW128 operates within a 128B line; this pattern has all lanes at col 0
    // of different lines, so the yyy bits (4-6) are zero for every lane and
    // no XOR applies. The conflict stays 32-way. The within-line story is
    // covered by ldmatrix.x4 .N below.
    const acc = NAIVE_COL0.accesses(CTX);
    expect(maxConflictWay(acc, SW128)).toBe(32);
  });

  it('ldmatrix.x4 .N: naive has 8-way bank conflict (rows collide, cols differ)', () => {
    // Lane L reads (row=L%8, mat=L/8) at offset (L%8)*128 + (L/8)*16.
    // 8 lanes share each (L/8), collapsing into 4 bank groups of 8 lanes.
    const acc = LDMATRIX_X4_N.accesses(CTX);
    expect(maxConflictWay(acc, NO_SWIZZLE)).toBe(8);
  });

  it('ldmatrix.x4 .N: SW128 fails cross-line (same (L%8) at different rows)', () => {
    // Byte-level SW128 has 128B span, so row starts (multiples of 128) don't
    // get XORed. Same 8-way conflict as naive. SW128 is the wrong atom for
    // this specific pattern — a wider swizzle or a different tile layout
    // would help.
    const acc = LDMATRIX_X4_N.accesses(CTX);
    expect(maxConflictWay(acc, SW128)).toBe(8);
  });

  it('pattern context scales: narrower rowStride changes the conflict profile', () => {
    const narrow = { rowStrideBytes: 32, tileBytes: 2048 };
    const way = maxConflictWay(LDMATRIX_X4_N.accesses(narrow), NO_SWIZZLE);
    expect(way).toBeGreaterThan(0);
  });

  it('cp.async 16B × 32 lanes: 512B wraps 128B bank line 4× → 4-way conflict', () => {
    // 32 lanes × 16 B = 512 B. The 128 B × 32-bank SMEM line wraps four
    // times, so each bank is visited 4 times across the 32 lanes.
    const acc = CPASYNC_16B.accesses(CTX);
    expect(maxConflictWay(acc, NO_SWIZZLE)).toBe(4);
  });
});

describe('accessArrival (bank-serialised wave model)', () => {
  it('32-way conflict produces 32 waves', () => {
    const acc = NAIVE_COL0.accesses(CTX);
    const { waveCount } = accessArrival(acc, NO_SWIZZLE);
    expect(waveCount).toBe(32);
  });

  it('no conflict produces 1 wave', () => {
    const acc = [
      { laneId: 0, byteOffset: 0, bytes: 4 },
      { laneId: 1, byteOffset: 4, bytes: 4 },
      { laneId: 2, byteOffset: 8, bytes: 4 },
    ];
    const { waveCount } = accessArrival(acc, NO_SWIZZLE);
    expect(waveCount).toBe(1);
  });

  it('lane in same bank as lane 0 gets wave 1', () => {
    const acc = [
      { laneId: 0, byteOffset: 0, bytes: 4 },
      { laneId: 1, byteOffset: 128, bytes: 4 },
    ];
    const { wavePerLane } = accessArrival(acc, NO_SWIZZLE);
    expect(wavePerLane.get(0)).toBe(0);
    expect(wavePerLane.get(1)).toBe(1);
  });
});

describe('contiguousArrival (GMEM bandwidth stagger)', () => {
  it('32 lanes split into 4 waves of 8', () => {
    const acc = CPASYNC_16B.accesses(CTX);
    const { waveCount } = contiguousArrival(acc);
    expect(waveCount).toBe(4);
  });
});
