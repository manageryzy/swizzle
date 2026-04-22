import { describe, expect, it } from 'vitest';
import {
  BANKS_PER_SUBPART,
  BYTES_PER_ENTRY,
  BYTES_PER_SM_RF,
  LRF_BANKS,
  LRF_ENTRIES_PER_BANK,
  SUBPARTITIONS,
  fragmentRegIds,
  readCycles,
  regToPhysical,
  subpartOfBank,
} from './rf';

describe('LRF geometry', () => {
  it('8 banks × 4 subpartitions = 2 banks per subpartition', () => {
    expect(LRF_BANKS).toBe(8);
    expect(SUBPARTITIONS).toBe(4);
    expect(BANKS_PER_SUBPART).toBe(2);
  });

  it('sizes multiply to the known 256 KiB Volta RF', () => {
    expect(BYTES_PER_ENTRY).toBe(128);
    expect(BYTES_PER_SM_RF).toBe(256 * 1024);
  });
});

describe('regToPhysical — even→low, odd→high bank of the pair', () => {
  it('subpartition 0 maps R0 to bank 0, R1 to bank 1', () => {
    expect(regToPhysical(0, 0)).toEqual({ bank: 0, entry: 0 });
    expect(regToPhysical(0, 1)).toEqual({ bank: 1, entry: 0 });
    expect(regToPhysical(0, 2)).toEqual({ bank: 0, entry: 1 });
    expect(regToPhysical(0, 3)).toEqual({ bank: 1, entry: 1 });
  });

  it('subpartition 2 uses banks 4 and 5', () => {
    expect(regToPhysical(2, 0).bank).toBe(4);
    expect(regToPhysical(2, 1).bank).toBe(5);
    expect(regToPhysical(2, 7).bank).toBe(5);
    expect(regToPhysical(2, 7).entry).toBe(3);
  });

  it('entry count saturates at LRF_ENTRIES_PER_BANK / 2 per warp', () => {
    const biggest = regToPhysical(0, 2 * LRF_ENTRIES_PER_BANK - 1);
    expect(biggest.entry).toBe(LRF_ENTRIES_PER_BANK - 1);
  });
});

describe('subpartOfBank is the inverse of the pair assignment', () => {
  it('bank 0,1 → subp 0; bank 6,7 → subp 3', () => {
    expect(subpartOfBank(0)).toBe(0);
    expect(subpartOfBank(1)).toBe(0);
    expect(subpartOfBank(6)).toBe(3);
    expect(subpartOfBank(7)).toBe(3);
  });
});

describe('readCycles — 1 read port per bank', () => {
  it('two sources in different banks read in 1 cycle', () => {
    expect(readCycles([0, 1])).toBe(1);
  });
  it('three sources with two in the same bank take 2 cycles', () => {
    // R0, R2 both in bank 0 (even); R1 in bank 1.
    expect(readCycles([0, 2, 1])).toBe(2);
  });
  it('four sources all in bank 0 take 4 cycles', () => {
    expect(readCycles([0, 2, 4, 6])).toBe(4);
  });
});

describe('fragmentRegIds', () => {
  it('returns a contiguous band of V registers', () => {
    expect(fragmentRegIds(4)).toEqual([0, 1, 2, 3]);
    expect(fragmentRegIds(4, 8)).toEqual([8, 9, 10, 11]);
  });
});
