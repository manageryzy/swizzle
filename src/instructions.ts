// Instruction catalog mirroring CUTLASS arch/ + cute/atom/mma_traits_sm*.hpp.
// Representative entries for M0 — expanded per-family in M1/M2/M3.

export type Arch = 'sm70' | 'sm75' | 'sm80' | 'sm89' | 'sm90' | 'sm100';
export type Family = 'wmma' | 'mma' | 'wgmma' | 'tcgen05' | 'tcgen05.block_scaled';

export type Dtype =
  | 'fp16'
  | 'bf16'
  | 'tf32'
  | 'fp32'
  | 'fp64'
  | 'fp8.e4m3'
  | 'fp8.e5m2'
  | 'fp6.e2m3'
  | 'fp6.e3m2'
  | 'fp4.e2m1'
  | 'mxfp4'
  | 'mxfp8'
  | 'mxfp4.nvfp4'
  | 'int8'
  | 'int4'
  | 'uint1';

export type OperandSource = 'smem' | 'rmem' | 'tmem' | 'gmem-wmma';
export type Major = 'K' | 'MN';

// Pipeline regime — drives whether producer (tma.load / cp.async) and consumer
// (mma step) execute in parallel warps (`warpspec`) or serially in the same
// warp (`coupled`). sm_90 wgmma and sm_100 tcgen05 kernels ship their
// CollectiveMma wrapped in `MainloopSm90Tma*` / `MainloopSm100Tma*` which
// uses two warp groups + mbarrier; sm_70 wmma and sm_80 mma do not.
export type PipelineMode = 'warpspec' | 'coupled';

// Whether a catalog entry is shipped as a canonical cute atom or is reachable
// only through the PTX-level shape rules. CUTLASS headers only expose 8 N
// values for wgmma (the `atom` subset); PTX §9.7.15.5.2 also allows every
// other multiple of 8 up to 256. The UI hides the PTX-only entries by
// default; a toggle in `ConfigBar` exposes them.
export type ShapeSource = 'cutlass-atom' | 'ptx-only';

export interface InstSpec {
  id: string;
  arch: Arch;
  family: Family;
  mnemonic: string;
  M: number;
  N: number;
  K: number;
  aDtypes: Dtype[];
  bDtypes: Dtype[];
  accDtypes: Dtype[];
  aSource: OperandSource[];
  bSource: OperandSource[];
  accIn: 'rmem' | 'tmem';
  majorA: Major[];
  majorB: Major[];
  ctaGroup?: 1 | 2;
  sparse?: boolean;
  warpSpecialized?: boolean;
  shapeSource?: ShapeSource;
  /** warpspec (producer∥consumer warps, mbarrier coord) or coupled (one warp, serial). */
  pipelineMode: PipelineMode;
  source: string; // reference to CUTLASS file
}

// CUTLASS ships canonical `SM90_64xNxK_*` atoms for these 8 N values only
// (`cute/arch/mma_sm90_gmma.hpp`). PTX allows any multiple of 8 from 8 to 256
// (32 shapes); the full set is `WGMMA_NS_PTX` below.
const WGMMA_NS_ATOM: readonly number[] = [8, 16, 32, 64, 96, 128, 192, 256];
const WGMMA_NS_PTX: readonly number[] = Array.from({ length: 32 }, (_, i) => (i + 1) * 8);
// TCGEN05_NS is the "CUTLASS subset" we render by default — PTX Table 41
// allows any N%8==0 for cg1 (and any N%16==0 for cg2) up to 256, giving 32
// shapes per kind. Staying with the 8-entry subset keeps the picker short;
// below we filter by N%16==0 for cg2 and restrict to {64,128,256} for .ws.
const TCGEN05_NS = [8, 16, 32, 64, 96, 128, 192, 256] as const;
const TCGEN05_NS_CG2 = TCGEN05_NS.filter((n) => n % 16 === 0); // [16,32,64,96,128,192,256]
const TCGEN05_NS_WS = [64, 128, 256] as const;

// wgmma.mma_async has distinct PTX mnemonics per acc dtype:
//   `.f32.f16.f16` (fp32 acc) vs `.f16.f16.f16` (fp16 acc) — see
//   PTX §9.7.15.5.2. CUTLASS `mma_sm90_gmma.hpp` ships separate MMA atoms for
//   each (SM90_64xNxK_F32F16F16_SS vs SM90_64xNxK_F16F16F16_SS). Split the
//   catalog so the mnemonic + C_LAYOUTS lookup reflect the chosen acc.
function wgmmaF16(N: number, acc: 'fp32' | 'fp16', shapeSource: ShapeSource): InstSpec {
  const accTag = acc === 'fp32' ? 'f32f16' : 'f16f16';
  const accMne = acc === 'fp32' ? 'f32' : 'f16';
  return {
    id: `sm90.wgmma.m64n${N}k16.${accTag}`,
    arch: 'sm90',
    family: 'wgmma',
    mnemonic: `wgmma.mma_async.sync.m64n${N}k16.${accMne}.f16.f16`,
    M: 64,
    N,
    K: 16,
    aDtypes: ['fp16', 'bf16'],
    bDtypes: ['fp16', 'bf16'],
    accDtypes: [acc],
    aSource: ['smem', 'rmem'],
    bSource: ['smem'],
    accIn: 'rmem',
    majorA: ['K', 'MN'],
    majorB: ['K', 'MN'],
    shapeSource,
    pipelineMode: 'warpspec',
    source: 'cute/arch/mma_sm90_gmma.hpp',
  };
}

// Tcgen05 "kinds" — each has a dtype set, a K derived from the dtype bit-width
// via cute's rule K = 256 / sizeof_bits<A> (or 512 / ... for mxf4nvf4).
interface TcgKind {
  name: string; // cute MMA family name (f16, tf32, i8, …)
  family: 'tcgen05' | 'tcgen05.block_scaled';
  ptxKind: string; // the literal .kind:: suffix in the mnemonic
  aDtypes: Dtype[];
  bDtypes: Dtype[];
  accDtypes: Dtype[];
  K: number;
}

const TCGEN05_KINDS: TcgKind[] = [
  { name: 'tf32', family: 'tcgen05', ptxKind: 'tf32',
    aDtypes: ['tf32'], bDtypes: ['tf32'], accDtypes: ['fp32'], K: 8 },
  { name: 'f16', family: 'tcgen05', ptxKind: 'f16',
    aDtypes: ['fp16', 'bf16'], bDtypes: ['fp16', 'bf16'], accDtypes: ['fp32'], K: 16 },
  { name: 'i8', family: 'tcgen05', ptxKind: 'i8',
    aDtypes: ['int8'], bDtypes: ['int8'], accDtypes: ['fp32'], K: 32 },
  { name: 'f8f6f4', family: 'tcgen05', ptxKind: 'f8f6f4',
    aDtypes: ['fp8.e4m3', 'fp8.e5m2', 'fp6.e2m3', 'fp6.e3m2', 'fp4.e2m1'],
    bDtypes: ['fp8.e4m3', 'fp8.e5m2', 'fp6.e2m3', 'fp6.e3m2', 'fp4.e2m1'],
    accDtypes: ['fp32'], K: 32 },
  { name: 'mxf8f6f4', family: 'tcgen05.block_scaled', ptxKind: 'mxf8f6f4.block_scale',
    aDtypes: ['mxfp8', 'fp8.e4m3', 'fp8.e5m2', 'fp6.e2m3', 'fp6.e3m2', 'fp4.e2m1'],
    bDtypes: ['mxfp8', 'fp8.e4m3', 'fp8.e5m2', 'fp6.e2m3', 'fp6.e3m2', 'fp4.e2m1'],
    accDtypes: ['fp32'], K: 32 },
  { name: 'mxf4nvf4', family: 'tcgen05.block_scaled', ptxKind: 'mxf4nvf4.block_scale',
    aDtypes: ['mxfp4.nvfp4'], bDtypes: ['mxfp4.nvfp4'], accDtypes: ['fp32'], K: 128 },
];

function tcgen05Mma(
  M: 32 | 64 | 128 | 256,
  N: number,
  K: number,
  cta: 1 | 2,
  kind: TcgKind,
  opts: { sparse?: boolean; ws?: boolean } = {},
): InstSpec {
  const sparse = !!opts.sparse;
  const ws = !!opts.ws;
  const suffix = `${sparse ? '.sp' : ''}${ws ? '.ws' : ''}`;
  return {
    id: `sm100.tcgen05.cg${cta}.m${M}n${N}k${K}.${kind.name}${suffix}`,
    arch: 'sm100',
    family: kind.family,
    mnemonic: `tcgen05.mma${sparse ? '.sp' : ''}${ws ? '.ws' : ''}.cta_group::${cta}.kind::${kind.ptxKind}.m${M}n${N}k${K}`,
    M,
    N,
    K,
    aDtypes: kind.aDtypes,
    bDtypes: kind.bDtypes,
    accDtypes: kind.accDtypes,
    aSource: ['smem', 'tmem'],
    bSource: ['smem'],
    accIn: 'tmem',
    majorA: ['K', 'MN'],
    majorB: ['K', 'MN'],
    ctaGroup: cta,
    sparse,
    warpSpecialized: ws,
    pipelineMode: 'warpspec',
    source: 'cute/arch/mma_sm100_umma.hpp',
  };
}

// Enumerate valid tcgen05 shapes for one kind per PTX ISA §9.7.16.2.1 Table 41.
//
// Table 41 rules (condensed):
//   f16 / tf32 / f8f6f4 / i8 (non-block-scaled):
//     cg1 dense   M ∈ {64,128},  N%8==0 in [8..256]
//     cg2 dense   M ∈ {128,256}, N%16==0 in [16..256]
//     cg1 sparse  same as cg1 dense
//     cg2 sparse  same as cg2 dense
//     .ws cg1     M ∈ {32,64,128}, N ∈ {64,128,256}, dense only
//     (no .ws cg2)
//   mxf8f6f4 (block-scaled):
//     cg1 dense   M=128,  N%8==0
//     cg2 dense   M ∈ {128,256}, N%16==0
//     cg2 sparse  M ∈ {128,256}, N%16==0
//     (no cg1 sparse, no .ws)
//   mxf4nvf4 (block-scaled):
//     cg1 dense   M=128,  N%8==0, K=64
//     cg2 dense   M ∈ {128,256}, N%16==0, K=64
//     cg2 sparse  M=256,  N%16==0, K=128
//     (no cg1 sparse, no .ws)
function tcgen05Instructions(kind: TcgKind): InstSpec[] {
  const out: InstSpec[] = [];
  const K = kind.K;
  const name = kind.name;

  if (name === 'f16' || name === 'tf32' || name === 'f8f6f4' || name === 'i8') {
    for (const M of [64, 128] as const)
      for (const N of TCGEN05_NS) out.push(tcgen05Mma(M, N, K, 1, kind));
    for (const M of [128, 256] as const)
      for (const N of TCGEN05_NS_CG2) out.push(tcgen05Mma(M, N, K, 2, kind));
    for (const M of [64, 128] as const)
      for (const N of TCGEN05_NS) out.push(tcgen05Mma(M, N, K, 1, kind, { sparse: true }));
    for (const M of [128, 256] as const)
      for (const N of TCGEN05_NS_CG2)
        out.push(tcgen05Mma(M, N, K, 2, kind, { sparse: true }));
    for (const M of [32, 64, 128] as const)
      for (const N of TCGEN05_NS_WS) out.push(tcgen05Mma(M, N, K, 1, kind, { ws: true }));
  } else if (name === 'mxf8f6f4') {
    for (const N of TCGEN05_NS) out.push(tcgen05Mma(128, N, K, 1, kind));
    for (const M of [128, 256] as const)
      for (const N of TCGEN05_NS_CG2) out.push(tcgen05Mma(M, N, K, 2, kind));
    for (const M of [128, 256] as const)
      for (const N of TCGEN05_NS_CG2)
        out.push(tcgen05Mma(M, N, K, 2, kind, { sparse: true }));
  } else if (name === 'mxf4nvf4') {
    for (const N of TCGEN05_NS) out.push(tcgen05Mma(128, N, 64, 1, kind));
    for (const M of [128, 256] as const)
      for (const N of TCGEN05_NS_CG2) out.push(tcgen05Mma(M, N, 64, 2, kind));
    for (const N of TCGEN05_NS_CG2)
      out.push(tcgen05Mma(256, N, 128, 2, kind, { sparse: true }));
  }
  return out;
}

export const INSTRUCTIONS: InstSpec[] = [
  // --- sm_70 wmma (nvcuda wmma intrinsics) ---
  {
    id: 'sm70.wmma.m16n16k16.f16',
    arch: 'sm70',
    family: 'wmma',
    mnemonic: 'wmma.mma.sync.m16n16k16.f16.f16.f16',
    M: 16,
    N: 16,
    K: 16,
    aDtypes: ['fp16'],
    bDtypes: ['fp16'],
    accDtypes: ['fp16', 'fp32'],
    aSource: ['gmem-wmma'],
    bSource: ['gmem-wmma'],
    accIn: 'rmem',
    majorA: ['K', 'MN'],
    majorB: ['K', 'MN'],
    pipelineMode: 'coupled',
    source: 'cutlass/arch/wmma_sm70.h',
  },
  {
    id: 'sm72.wmma.m16n16k16.s8',
    arch: 'sm75',
    family: 'wmma',
    mnemonic: 'wmma.mma.sync.m16n16k16.s32.s8.s8.s32',
    M: 16,
    N: 16,
    K: 16,
    aDtypes: ['int8'],
    bDtypes: ['int8'],
    accDtypes: ['fp32'],
    aSource: ['gmem-wmma'],
    bSource: ['gmem-wmma'],
    accIn: 'rmem',
    majorA: ['K'],
    majorB: ['K'],
    pipelineMode: 'coupled',
    source: 'cutlass/arch/wmma_sm72.h',
  },

  // --- sm_80 mma (register-sourced) ---
  {
    id: 'sm80.mma.m16n8k16.f16',
    arch: 'sm80',
    family: 'mma',
    mnemonic: 'mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32',
    M: 16,
    N: 8,
    K: 16,
    aDtypes: ['fp16', 'bf16'],
    bDtypes: ['fp16', 'bf16'],
    accDtypes: ['fp32', 'fp16'],
    aSource: ['rmem'],
    bSource: ['rmem'],
    accIn: 'rmem',
    majorA: ['K'],
    majorB: ['MN'],
    pipelineMode: 'coupled',
    source: 'cute/atom/mma_traits_sm80.hpp',
  },
  {
    id: 'sm80.mma.m16n8k8.tf32',
    arch: 'sm80',
    family: 'mma',
    mnemonic: 'mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32',
    M: 16,
    N: 8,
    K: 8,
    aDtypes: ['tf32'],
    bDtypes: ['tf32'],
    accDtypes: ['fp32'],
    aSource: ['rmem'],
    bSource: ['rmem'],
    accIn: 'rmem',
    majorA: ['K'],
    majorB: ['MN'],
    pipelineMode: 'coupled',
    source: 'cute/atom/mma_traits_sm80.hpp',
  },
  {
    id: 'sm89.mma.m16n8k32.fp8',
    arch: 'sm89',
    family: 'mma',
    mnemonic: 'mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32',
    M: 16,
    N: 8,
    K: 32,
    aDtypes: ['fp8.e4m3', 'fp8.e5m2'],
    bDtypes: ['fp8.e4m3', 'fp8.e5m2'],
    accDtypes: ['fp32'],
    aSource: ['rmem'],
    bSource: ['rmem'],
    accIn: 'rmem',
    majorA: ['K'],
    majorB: ['MN'],
    pipelineMode: 'coupled',
    source: 'cute/atom/mma_traits_sm89.hpp',
  },

  // --- sm_90 wgmma (SS + RS) — one entry per (N, acc dtype, shapeSource) ---
  ...WGMMA_NS_PTX.flatMap((n) => {
    const isAtom = WGMMA_NS_ATOM.includes(n);
    const src: ShapeSource = isAtom ? 'cutlass-atom' : 'ptx-only';
    return [wgmmaF16(n, 'fp32', src), wgmmaF16(n, 'fp16', src)];
  }),

  // --- sm_100 tcgen05: all kinds × M × N × cta_group ---
  //
  // Emitted via `tcgen05Instructions(kind)` so the (M, cg, sparse, ws) matrix
  // tracks PTX ISA §9.7.16.2.1 Table 41 rather than a hand-picked subset.
  // Layouts per (M, cta_group):
  //   M=32 cg1 .ws → G ·  M=64 cg1 non-ws → F ·  M=64 cg1 .ws → E
  //   M=128 cg1 → D    ·  M=128 cg2 dense → B ·  M=128 cg2 sparse → C
  //   M=256 cg2 → A
  ...TCGEN05_KINDS.flatMap((kind) => tcgen05Instructions(kind)),
];

export function findInst(id: string): InstSpec | undefined {
  return INSTRUCTIONS.find((i) => i.id === id);
}

export function instsForArch(arch: Arch): InstSpec[] {
  return INSTRUCTIONS.filter((i) => i.arch === arch);
}
