// SMEM budget model — mirrors CUTLASS `compute_stage_count_or_override` for
// the TMA-warpspecialized mainloop. Ground truth:
//   /tmp/cutlass/include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl:71
//
//   stage_bytes = round_up(bits_to_bytes(a_bits * M * K)
//                          + bits_to_bytes(b_bits * N * K), 128)
//                 + sizeof(PipelineTmaAsync<1>::SharedStorage)
//   kStages = (capacity - carveout) / stage_bytes
//
// Key terminology (distinct concepts often confused):
//   * kStages  — compile-time pipeline depth = # of SMEM ring buffers for A+B.
//   * tileK    — how much K this CTA iterates over per output tile.
//   * K iters  — tileK / kAtomK, how many times the mma instruction fires.
//
// kStages and tileK are ORTHOGONAL. Pipeline stages rotate modulo kStages as
// each K iteration fires.
//
// Per-SM SMEM whitepaper numbers:
//   sm_70 96 KiB · sm_75 64 · sm_80 164 · sm_89 100 · sm_90 228 · sm_100 228
//
// mainloop_pipeline_bytes: for PipelineTmaAsync this is ~2 mbarriers per stage
// (full_barrier + empty_barrier, 8 B each) + a small pipeline state. Rounded
// to 32 B/stage for honesty — the real figure depends on pipeline class.

import type { Arch, Dtype, InstSpec } from './instructions';

export const SMEM_PER_SM: Record<Arch, number> = {
  sm70: 96 * 1024,
  sm75: 64 * 1024,
  sm80: 164 * 1024,
  sm89: 100 * 1024,
  sm90: 228 * 1024,
  sm100: 228 * 1024,
};

export const ALIGN = 128;
export const MAINLOOP_PIPELINE_BYTES_PER_STAGE = 32;

export function bytesOf(dtype: Dtype): number {
  switch (dtype) {
    case 'fp64': return 8;
    case 'fp32':
    case 'tf32': return 4;
    case 'fp16':
    case 'bf16': return 2;
    case 'fp8.e4m3':
    case 'fp8.e5m2':
    case 'mxfp8':
    case 'int8': return 1;
    case 'fp6.e2m3':
    case 'fp6.e3m2': return 0.75;
    case 'fp4.e2m1':
    case 'mxfp4':
    case 'mxfp4.nvfp4': return 0.5;
    case 'int4': return 0.5;
    case 'uint1': return 0.125;
  }
}

export function alignUp(n: number, align = ALIGN): number {
  return Math.ceil(n / align) * align;
}

export function tileBytesRaw(rows: number, cols: number, dtype: Dtype): number {
  return Math.ceil(rows * cols * bytesOf(dtype));
}

// Exposed for the UI: size of one A+B pair, aligned to 128, + mbar overhead.
export function stageBytes(i: InstSpec): {
  a: number;
  b: number;
  pair: number;     // round_up(a+b, 128)
  pad: number;      // pair - a - b
  mbar: number;
  stage: number;    // pair + mbar, as the builder uses
} {
  const dtypeA = i.aDtypes[0];
  const dtypeB = i.bDtypes[0];
  const a = tileBytesRaw(i.M, i.K, dtypeA);
  const b = tileBytesRaw(i.N, i.K, dtypeB);
  const pair = alignUp(a + b);
  const pad = pair - a - b;
  const mbar = MAINLOOP_PIPELINE_BYTES_PER_STAGE;
  return { a, b, pair, pad, mbar, stage: pair + mbar };
}

export type Segment = {
  kind: 'A' | 'B' | 'pad' | 'mbar' | 'unused';
  stage?: number;
  bytes: number;
  label: string;
};

export interface Budget {
  arch: Arch;
  total: number;
  stages: number;
  stageBytes: ReturnType<typeof stageBytes>;
  segments: Segment[];
  usedBytes: number;
  fits: boolean;
}

export function buildBudget(i: InstSpec, stages: number): Budget {
  const total = SMEM_PER_SM[i.arch];
  const sb = stageBytes(i);

  const segments: Segment[] = [];
  for (let s = 0; s < stages; s++) {
    segments.push({ kind: 'A', stage: s, bytes: sb.a, label: `A[${s}] ${sb.a}B` });
    segments.push({ kind: 'B', stage: s, bytes: sb.b, label: `B[${s}] ${sb.b}B` });
    if (sb.pad > 0)
      segments.push({ kind: 'pad', stage: s, bytes: sb.pad, label: `pad ${sb.pad}B` });
    segments.push({ kind: 'mbar', stage: s, bytes: sb.mbar, label: `mbar ${sb.mbar}B` });
  }

  const usedBytes = stages * sb.stage;
  const unused = total - usedBytes;
  if (unused > 0)
    segments.push({ kind: 'unused', bytes: unused, label: `free ${(unused / 1024).toFixed(1)}K` });

  return { arch: i.arch, total, stages, stageBytes: sb, segments, usedBytes, fits: usedBytes <= total };
}

// Recommended max stages that fit in SMEM. Mirrors the CUTLASS AutoCarveout
// division: (capacity - carveout) / stage_bytes. carveout=0 baseline.
export function maxStages(i: InstSpec, carveoutBytes = 0): number {
  const sb = stageBytes(i);
  const capacity = SMEM_PER_SM[i.arch];
  if (sb.stage <= 0) return 1;
  return Math.max(1, Math.floor((capacity - carveoutBytes) / sb.stage));
}

// K iteration count for a given CTA tile size. `tileK` is the total K the CTA
// will sweep over per output tile; each iteration fires one mma atom.
export function kIterations(i: InstSpec, tileK: number): number {
  return Math.max(1, Math.ceil(tileK / i.K));
}
