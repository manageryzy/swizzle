// Access patterns modelled as warp-level {lane → byte offset, bytes} sets.
// The visualization picks a pattern per phase and renders lane overlays on
// top of the SMEM bank heatmap.
//
// Ground truth for ldmatrix.x4 .N / .T comes from
//   /tmp/cutlass/include/cute/atom/copy_traits_sm75.hpp (SM75_U32x4_LDSM_N/T)
//   PTX ISA §9.7.14 "ldmatrix".

import { apply, bankOfByte, type Swizzle } from './swizzle';

export interface Access {
  laneId: number;
  byteOffset: number;
  bytes: number; // 4 for 32b, 16 for 128b (ldmatrix)
}

// Patterns produce accesses relative to the tile geometry. `rowStrideBytes`
// is the per-row byte stride; patterns wrap with modulo so the same pattern
// stays in-bounds for any tile size.
export interface PatternContext {
  rowStrideBytes: number;
  tileBytes: number;
}

export interface Pattern {
  id: string;
  name: string;
  description: string;
  bytesPerLane: number;
  accesses: (ctx: PatternContext) => Access[];
}

export const DEFAULT_CTX: PatternContext = {
  rowStrideBytes: 128,
  tileBytes: 4096,
};

function wrap(off: number, size: number): number {
  return size > 0 ? off % size : off;
}

// 32 lanes, each reads col 0 of a different row. Offset = lane * rowStride,
// wrapped to tile size.
export const NAIVE_COL0: Pattern = {
  id: 'naive.col0',
  name: 'naive col-0 × 32 rows',
  description:
    '32 lanes each read (row=lane, col=0). All lanes hit column 0 → every lane lands in the same bank word at row r*rowStride → 32-way conflict.',
  bytesPerLane: 4,
  accesses: (ctx) =>
    Array.from({ length: 32 }, (_, L) => ({
      laneId: L,
      byteOffset: wrap(L * ctx.rowStrideBytes, ctx.tileBytes),
      bytes: 4,
    })),
};

// ldmatrix.x4 .N — warp cooperatively loads 4 × 8×8 fp16 matrices.
// Lane L provides the SMEM row address for row (L%8) of matrix (L/8).
// Each matrix row is 16 B; the tile rowStride is whatever the real inst has.
export const LDMATRIX_X4_N: Pattern = {
  id: 'ldmatrix.x4.N',
  name: 'ldmatrix.x4 .N (fp16 K-major)',
  description:
    'Canonical warp-cooperative load of 4×(8×8 fp16) matrices. Lane L supplies the row pointer for row (L%8) of matrix (L/8). Without swizzle: heavy bank conflict because all 8 row starts land in the same bank group.',
  bytesPerLane: 16,
  accesses: (ctx) =>
    Array.from({ length: 32 }, (_, L) => ({
      laneId: L,
      byteOffset: wrap((L % 8) * ctx.rowStrideBytes + Math.floor(L / 8) * 16, ctx.tileBytes),
      bytes: 16,
    })),
};

// ldmatrix.x4 .T — transposed variant. Lane L provides addr for col-group.
export const LDMATRIX_X4_T: Pattern = {
  id: 'ldmatrix.x4.T',
  name: 'ldmatrix.x4 .T (MN-major, transposed)',
  description:
    'Transposed variant. Lane L reads a column-major slice; naive layout collides hard, SW64/SW128 restore throughput.',
  bytesPerLane: 4,
  accesses: (ctx) =>
    Array.from({ length: 32 }, (_, L) => {
      const row = Math.floor(L / 4) % 8;
      const mat = Math.floor(L / 8);
      return {
        laneId: L,
        byteOffset: wrap(row * ctx.rowStrideBytes + mat * 16 + (L % 4) * 4, ctx.tileBytes),
        bytes: 4,
      };
    }),
};

// cp.async 16B per lane, contiguous. Tends to wrap across rows for tiles
// smaller than 512B.
export const CPASYNC_16B: Pattern = {
  id: 'cp.async.16B',
  name: 'cp.async 16B × 32 lanes',
  description:
    'Each lane issues a 16-byte cp.async. Contiguous byte offsets with 32-lane-linear stride. For tiles ≤ 128B the accesses wrap and collide.',
  bytesPerLane: 16,
  accesses: (ctx) =>
    Array.from({ length: 32 }, (_, L) => ({
      laneId: L,
      byteOffset: wrap(L * 16, ctx.tileBytes),
      bytes: 16,
    })),
};

export const PATTERNS: Record<string, Pattern> = {
  [NAIVE_COL0.id]: NAIVE_COL0,
  [LDMATRIX_X4_N.id]: LDMATRIX_X4_N,
  [LDMATRIX_X4_T.id]: LDMATRIX_X4_T,
  [CPASYNC_16B.id]: CPASYNC_16B,
};

export interface Conflict {
  bank: number;
  lanes: number[];
  way: number; // distinct lanes on this bank in a single cycle
}

export function analyzeConflicts(accesses: Access[], sw: Swizzle): Conflict[] {
  // Each access contributes one entry per 4-byte word touched.
  const perBank = new Map<number, Set<number>>();
  for (const a of accesses) {
    const physBase = apply(sw, a.byteOffset);
    for (let b = 0; b < a.bytes; b += 4) {
      const bank = bankOfByte(physBase + b);
      const s = perBank.get(bank) ?? new Set<number>();
      s.add(a.laneId);
      perBank.set(bank, s);
    }
  }
  const out: Conflict[] = [];
  for (const [bank, lanes] of perBank) {
    if (lanes.size > 1) out.push({ bank, lanes: [...lanes].sort((x, y) => x - y), way: lanes.size });
  }
  out.sort((a, b) => b.way - a.way);
  return out;
}

export function maxConflictWay(accesses: Access[], sw: Swizzle): number {
  const conflicts = analyzeConflicts(accesses, sw);
  return conflicts.length === 0 ? 1 : conflicts[0].way;
}

// Bank-serialised arrival schedule. For each lane's access, compute the wave
// (bank-conflict cycle) at which it completes: the max position-in-queue over
// the banks it touches. Lanes in the same wave arrive in parallel.
//
// Returns `{ wavePerLane, waveCount }`. waveCount = maxConflictWay(). Used by
// SmemPanel to fade-in each lane ring at the right sub-progress.
export function accessArrival(
  accesses: Access[],
  sw: Swizzle,
): { wavePerLane: Map<number, number>; waveCount: number } {
  // Build each bank's arrival order.
  const perBank = new Map<number, number[]>();
  for (const a of accesses) {
    const phys = apply(sw, a.byteOffset);
    for (let b = 0; b < a.bytes; b += 4) {
      const bank = bankOfByte(phys + b);
      const list = perBank.get(bank);
      if (!list) perBank.set(bank, [a.laneId]);
      else if (!list.includes(a.laneId)) list.push(a.laneId);
    }
  }

  const wavePerLane = new Map<number, number>();
  for (const a of accesses) {
    const phys = apply(sw, a.byteOffset);
    let maxWave = 0;
    for (let b = 0; b < a.bytes; b += 4) {
      const bank = bankOfByte(phys + b);
      const pos = perBank.get(bank)!.indexOf(a.laneId);
      if (pos > maxWave) maxWave = pos;
    }
    wavePerLane.set(a.laneId, maxWave);
  }
  const waveCount = Math.max(0, ...wavePerLane.values()) + 1;
  return { wavePerLane, waveCount };
}

// Contiguous stagger (for cp.async / TMA where the bottleneck is GMEM
// bandwidth, not SMEM banks). Returns a per-lane wave = laneId // 8 so we see
// 4 waves of 8 lanes each.
export function contiguousArrival(accesses: Access[]): {
  wavePerLane: Map<number, number>;
  waveCount: number;
} {
  const wavePerLane = new Map<number, number>();
  let max = 0;
  for (const a of accesses) {
    const wave = a.laneId >> 3;
    wavePerLane.set(a.laneId, wave);
    if (wave > max) max = wave;
  }
  return { wavePerLane, waveCount: max + 1 };
}
