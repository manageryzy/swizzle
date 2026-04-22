// Main-SM register file (LRF) model, ported from the Volta VTS knobs:
//   LRFBanks = 8, LRFRegsPerBank = 256, numRFRdPorts = 1
// Organised into 4 subpartitions (subcores); each subpartition owns 2 of
// the 8 banks, so a warp in-flight has 2 banks and can deliver 2 reads per
// cycle. A 3-source FMA stalls one extra cycle in the collector stage.
//
// The per-arch geometry may drift (Blackwell trims RF/SM), but the 8-banks-
// × 4-subpart shape has held from Volta onward so the same visualisation
// applies to sm_70..sm_100.

export const LRF_BANKS = 8;
export const LRF_ENTRIES_PER_BANK = 256;
export const SUBPARTITIONS = 4;
export const BANKS_PER_SUBPART = LRF_BANKS / SUBPARTITIONS;
export const LANES_PER_WARP = 32;
export const BYTES_PER_LANE = 4;
export const BYTES_PER_ENTRY = LANES_PER_WARP * BYTES_PER_LANE; // 128 B
const BYTES_PER_BANK = LRF_ENTRIES_PER_BANK * BYTES_PER_ENTRY; // 32 KiB
export const BYTES_PER_SM_RF = LRF_BANKS * BYTES_PER_BANK; // 256 KiB

export interface PhysReg {
  bank: number; // 0..7
  entry: number; // 0..255
}

// Map logical register R_n of a warp running in subpartition s to its
// physical (bank, entry). Simplified teaching model — the actual hardware
// mapping is not exposed, but this reproduces the two-banks-per-warp,
// even/odd split used by CUDA register allocators.
export function regToPhysical(subpart: number, regId: number): PhysReg {
  const bank = subpart * BANKS_PER_SUBPART + (regId & 1);
  const entry = regId >>> 1;
  return { bank, entry };
}

// Inverse: which warps (subpartitions) can serve register accesses from a
// given bank? Exactly one subpartition per bank pair.
export function subpartOfBank(bank: number): number {
  return Math.floor(bank / BANKS_PER_SUBPART);
}

// Read-bank-conflict analysis: given a set of source regs an instruction
// reads in one cycle, return the number of cycles the collector stage
// needs. With 2 banks per warp and 1 read port per bank, parallel reads
// cap at 2 per cycle. Additional sources to the same bank serialise.
export function readCycles(sourceRegs: number[]): number {
  const perBank = new Map<number, number>();
  for (const r of sourceRegs) {
    const b = r & 1;
    perBank.set(b, (perBank.get(b) ?? 0) + 1);
  }
  return Math.max(1, ...perBank.values());
}

// For a CLayout with `valuesPerThread` regs, return the register ids each
// lane uses. The cute convention allocates a contiguous band R_base..R_base+V.
export function fragmentRegIds(valuesPerThread: number, base = 0): number[] {
  return Array.from({ length: valuesPerThread }, (_, i) => base + i);
}
