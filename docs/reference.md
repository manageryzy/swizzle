# Reference — ground truth extracted from CUTLASS + PTX ISA

## 0. Terminology note: "RMEM" vs `.reg`

The PTX ISA has no "RMEM" state space. Its state spaces are `.reg`,
`.shared`, `.global`, `.local`, `.const`, `.param`, and (sm_100) `.tensor`.

"RMEM" is a cute/CUTLASS coinage — from `cute/pointer.hpp:226`:

```cpp
// Anything that is not gmem or smem is rmem
template <class T>
struct is_rmem : bool_constant<not (is_gmem<T>::value || is_smem<T>::value)> {};
```

So `rmem` in cute means `.reg ∪ .local`. The demo's "Registers (.reg)" panel
is more literally correct; the string `'rmem'` survives in code to match the
CUTLASS token naming, and `RS` / `SS` operand-source terminology follows cute.

Everything in the demo derives from this file. When you touch a module, you are
expected to cite a line here or update this file first.

## 1. cute::Swizzle<B,M,S>

From `/tmp/cutlass/include/cute/swizzle.hpp`:

```cpp
// Template: Swizzle<BBits, MBase, SShift>
// num_bits = B; num_base = M; num_shft = S
// bit_msk = (1 << B) - 1
// yyy_msk = bit_msk << (M + max(0,S))
// zzz_msk = bit_msk << (M - min(0,S))
// apply(offset) = offset XOR shiftr(offset & yyy_msk, msk_sft)
```

- **B** — how many bits the XOR mask spans.
- **M** — base position; bits `[0, M)` are never touched. Controls the element
  granularity (byte lane within a 128-bit word, 16-bit lane within fp16, etc.).
- **S** — distance between the "source" Y bits and the "destination" Z bits they
  XOR into. The same Swizzle is its own inverse.

### Canonical atoms (from `mma_traits_sm90_gmma.hpp` / `mma_traits_sm100.hpp`)

| name | (B,M,S) | byte span | LayoutType id |
|---|---|---|---|
| INTER / no swizzle | `Swizzle<0,4,3>` | — | 0 |
| 32B | `Swizzle<1,4,3>` | 32 B | 6 |
| 64B | `Swizzle<2,4,3>` | 64 B | 4 |
| 128B | `Swizzle<3,4,3>` | 128 B | 2 |
| 128B base32B (sm_100) | `Swizzle<2,5,2>` | 128 B | 1 |

Composition with a Layout uses `cute::composition(sxor, offset, layout)` →
`ComposedLayout`. The swizzle wraps the *codomain* (physical offset), not the
input coordinate: `phys = sxor( offset + B(coord) )`.

### Upcast rule

`upcast<sizeof_bits<T>>(Swizzle<B,M,S>) = Swizzle<B, M - log2(sizeof_bits<T>), S>`
— i.e. when the tile carries a wider element, the "element-lane" base shrinks.
`Swizzle<3,4,3>` at bit level becomes `Swizzle<3,0,3>` for fp16 and fp32 alike
at element level (both drop M by 4 bits).

## 2. SmemDescriptor (`mma_sm100_desc.hpp`)

64-bit descriptor consumed by `wgmma.mma_async` and `tcgen05.mma`:

| bits | field | notes |
|---|---|---|
| [0, 14) | `start_address` | SMEM byte addr, 4 LSB dropped |
| [14, 16) | — | reserved |
| [16, 30) | `leading_byte_offset` | in 16-byte units |
| [30, 32) | — | reserved |
| [32, 46) | `stride_byte_offset` | in 16-byte units |
| [46, 48) | `version` | sm_100 = 1 |
| [49, 52) | `base_offset` | 3 bits |
| [52, 53) | `lbo_mode` | leading-byte-offset mode |
| [53, 61) | — | reserved |
| [61, 64) | `layout_type` | `LayoutType` enum — see above |

Builder: `cute::UMMA::make_umma_desc<Major::K|MN>(tensor)`.

## 3. TMEM geometry (sm_100)

From `/tmp/cutlass/include/cute/pointer.hpp` and `arch/tmem_allocator_sm100.hpp`:

- **128 DP lanes × 512 columns**, each cell 32 bits.
- 32-bit TMEM address: `[col:9 | dp:7 | idx:8 (debug)]`. Low 16 bits are
  `col | (dp<<9)`; top 8 hold a subword index for narrower types.
- DP stride in bits = `1 << 21`.
- Allocation granularity: 32 columns, pow-of-2 up to 512.
- `tcgen05.alloc / dealloc / relinquish_alloc_permit` manage per-SM capacity.

### tcgen05.ld / tcgen05.st shapes

From `cute/atom/copy_traits_sm100.hpp`:

- `SM100_TMEM_LOAD_16dp256b{1,2,4,8,16,32}x`
- `SM100_TMEM_LOAD_16dp128b{1,2,4,8}x`

16 DP lanes × {128b, 256b} × {1, 2, 4, 8, 16, 32} repeats. Example
`16dp256b1x`: 32 threads × 4096 bits, reading 256 bits per lane across 16 DP.

## 4. tcgen05.mma instruction family

From `cute/arch/mma_sm100_umma.hpp`:

- M ∈ {64, 128, 256}; 256 requires `cta_group::2`.
- N ∈ {8, 16, 32, 64, 96, 128, 192, 256}.
- K depends on kind: 16 (f16/tf32), 32 (i8, f8f6f4, mxf8f6f4), 64 (mxf4nvf4).
- Kinds: `tf32`, `f16`, `i8`, `f8f6f4`, `mxf8f6f4.block_scale`, `mxf4nvf4.block_scale`.
- Operand A: SMEM (SS variants) or TMEM (TS variants). Operand B: SMEM.
- Accumulator (D/C): TMEM. Scaled by `cta_group::1|2`.
- Synchronisation: `tcgen05.fence`, `tcgen05.commit`, `tcgen05.wait`.

## 5. wgmma.mma_async (sm_90)

From `cute/arch/mma_sm90_gmma.hpp`:

- `wgmma.m64n{8,16,32,64,96,128,192,256}k{16,32}` per kind.
- Issued by a warpgroup (128 threads). A is SMEM descriptor or register fragment
  (SS vs RS); B is always SMEM descriptor.
- Accumulator: RMEM (`uint32_t[acc_size]`).
- Sync: `wgmma.fence.sync.aligned`, `wgmma.commit_group`, `wgmma.wait_group<N>`.

## 6. sm_80 mma.m16n8k* canonical fragment (fp16, 16x8x16)

From `cute/atom/mma_traits_sm80.hpp`:

- `ALayout` = `Layout<Shape<Shape<_4,_8>,Shape<_2,_2,_2>>, Stride<Stride<_32,_1>,Stride<_16,_8,_128>>>`
- `BLayout` = `Layout<Shape<Shape<_4,_8>,Shape<_2,_2>>, Stride<Stride<_16,_1>,Stride<_8,_64>>>`
- `CLayout` = `Layout<Shape<Shape<_4,_8>,Shape<_2,_2>>, Stride<Stride<_32,_1>,Stride<_16,_8>>>`

## 7. wmma (sm_70 / sm_72 / sm_75)

Via `nvcuda::wmma` intrinsics, shapes 16x16x16 / 8x32x16 / 32x8x16. Dtypes:
fp16 (sm_70), int8 (sm_72), int4 / uint1 (sm_75). Source: `cutlass/arch/wmma_sm{70,72,75}.h`.

## 8. CollectiveBuilder dispatch (`gemm/collective/collective_builder.hpp`)

`CollectiveBuilder<ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, ElementAcc, TileShape, ClusterShape, StageCount, KernelSchedule>`

- `ArchTag` = `arch::Sm70 | Sm80 | Sm89 | Sm90 | Sm100 | Sm120`
- `KernelSchedule` selects between warp-specialized / pingpong / cooperative and
  TMA vs cp.async loaders.
- Per-arch mainloops live under `gemm/collective/builders/sm{90,100,103,120}_*.inl`.
- The builder SFINAE's on `(ArchTag, KernelSchedule, operand dtypes, major-ness)`
  to pick one `sm*_mma_warpspecialized_*.hpp` collective class.

This is the pattern the demo's `ConfigBar` mirrors: the user picks an arch +
instruction + schedule, and the panels derive the collective that would be
selected.

## 9. Tile hierarchy (what the `TileHierarchyPanel` visualises)

Every CUTLASS GEMM is a composition of four tile levels:

1. **Problem shape** `(M, N, K)` — the whole matmul `C += A · B`.
2. **ClusterShape** `(CLUSTER_M, CLUSTER_N, CLUSTER_K)` — the grid of cooperating CTAs. For sm_100 tcgen05 `cta_group::2` this is `(2,1,1)`; for plain sm_90 it is `(1,1,1)`.
3. **TileShape** `(BLK_M, BLK_N, BLK_K)` — the M/N/K covered by a single CTA.
   - Each CTA's output tile is `BLK_M × BLK_N` of `C`.
   - The CTA walks `numIters = ceil(K / BLK_K)` steps along K, producing a new K-slice of A (`BLK_M × BLK_K`) and B (`BLK_K × BLK_N`) per iter.
   - SMEM holds `PIPE = kStages` ring slots of those slices.
4. **Atom shape** `(atom_M, atom_N, atom_K)` — the MMA instruction (`wgmma.m64n128k16`, `tcgen05.mma.m128n128k16`, ...).

CUTLASS `TiledMMA<AtomLayoutMNK>` fans the atoms across the CTA tile. The
demo models `AtomLayoutMNK = (blkMMult, blkNMult, 1)` — integer multiples
only. The `TILE` selectors in ConfigBar set these multipliers.

- `Problem` → not drawn at scale; the TileHierarchyPanel Row 1 uses a 4×4
  schematic to show one CTA's position relative to its peers.
- `ClusterShape` → Row 1 outlines neighboring CTAs in the same cluster.
- `TileShape` → Row 2 shows each K slot as a column; Row 3 shows the CTA
  tile subdivided into atoms.
- `Atom` → one atom cell in Row 3; the currently-executing atom is yellow.
- `PIPE` → Row 2 slot colour: consume (yellow) / in-flight fill (green) /
  hold / drained.

Key CUTLASS APIs the panel maps to:
- `cute::local_tile(mA, make_shape(BLK_M, BLK_K), tile_coord)` — carves the
  CTA tile out of the GMEM A tensor.
- `cute::TiledMMA<AtomLayoutMNK, ValLayoutMNK>` — defines how atoms fan out
  to produce one CTA output tile. See `cute/atom/atom.hpp`.
- `cutlass::gemm::collective::StageCountAuto` — computes `kStages = PIPE`
  from the SMEM budget formula modelled in `src/smem_budget.ts`.

Naming map (UI ↔ CUTLASS):

| UI signal / label | CUTLASS name |
|---|---|
| `InstSpec.M, .N, .K` | atom shape `(atom_M, atom_N, atom_K)` |
| `blkMMult, blkNMult` | `AtomLayoutMNK_M`, `AtomLayoutMNK_N` |
| derived `BLK_M = InstSpec.M × blkMMult` | `TileShape_M` |
| derived `BLK_N = InstSpec.N × blkNMult` | `TileShape_N` |
| `tileK` | `TileShape_K` aka `BLK_K` |
| `kStages` | `PIPE` (aka `StageCount`) |
| `ctaGroup` (tcgen05 flag) | `ClusterShape_M` (when = 2) |
| `problemMMult, problemNMult, problemKMult` | problem-to-CTA-tile multiplier (shapes the GMEM rectangle) |
| `ctaGrid.rowsM/colsN/slicesK` | `cute::ceil_div(Problem, BLK)` grid dimensions |

## 10. Pipeline regimes (warpspec vs coupled)

NVIDIA GPUs have two distinct matmul pipeline shapes, and v3 of the
visualizer distinguishes them explicitly. Every `InstSpec` carries
`pipelineMode: 'warpspec' | 'coupled'`.

### `warpspec` — sm_90 wgmma, sm_100 tcgen05

One warpgroup (4 warps, 128 threads) is the **producer**: it only issues
`cp.async.bulk.tensor` (TMA). A different warpgroup is the **consumer**: it
only issues `wgmma.mma_async` (sm_90) or `tcgen05.mma` (sm_100). They
coordinate through an `mbarrier` ring (`PipelineTmaAsync<PIPE>` in
`cutlass/pipeline/sm90_pipeline.hpp`):

```
producer: producer_acquire(stage) → tma_load(sA(_,_,_,stage)) → mbar.arrive(stage)
consumer: mbar.wait(stage) → wgmma(sA(_,_,_,stage)) → ++stage
```

The two streams execute **concurrently in wall-clock time**. Once the ring
is primed, the producer is `PIPE − 1` K-slices ahead of the consumer. This
overlap is the reason CUTLASS kernels reach close to peak throughput on
Hopper/Blackwell, and is modelled in `pipeline_state.ts · emitTimeline()`
as two non-serialized `TimelinePhase[]` arrays whose `startTick`/`endTick`
ranges overlap.

CUTLASS wrappers that set this regime: `MainloopSm90TmaGmmaWarpSpecialized`,
`MainloopSm100TmaUmmaWarpSpecialized*`.

### `coupled` — sm_70 wmma, sm_80 mma.sync

A single warp issues the whole chain: `cp.async` (or `ld.global`) → `ldmatrix`
→ `mma.sync` → repeat. No mbarrier, no ring depth beyond 1, no parallelism
between load and compute. `emitTimeline()` produces strictly serial phases
in this regime: for each K slice, `producer.endTick ≤ consumer.startTick`.

### What the UI shows

- **ConfigBar** right-side pill: teal `ASYNC · producer ∥ consumer` for
  warpspec, gray `SYNC · same warp` for coupled.
- **Timeline** swim lanes: under warpspec the producer and consumer bars
  overlap on the tick axis; under coupled they tessellate end-to-end.
  The mbarrier row only populates for warpspec.
- **TileHierarchyPanel Row 2**: warpspec shows **two cursors** (solid
  consumer, dotted producer) walking K at different rates; coupled shows
  a single cursor.
- **MemFlowPanel**: warpspec can light multiple edges simultaneously
  (producer `tmaA` + consumer `descA` at the same tick); coupled lights
  one at a time.

## 11. GMEM ↔ SMEM under `Swizzle<B,M,S>` (what `GmemPanel` visualises)

The GmemPanel sits above SmemPanel so the reader sees the full story:
GMEM tile → SMEM line (under swizzle) → SMEM banks → warp fragment.

Three tracks per operand (A, B) plus a C-tile destination track:

1. **GMEM tile** — the M×K (A) or K×N (B) rectangle gridded at CTA-tile
   boundaries. "This CTA" at (0,0) is highlighted; rainbow stripes color
   each logical SMEM line of the currently-loading K slab.
2. **Arrows** — one arrow per logical SMEM line, pointing at its physical
   line index under the active `Swizzle<B,M,S>`. The arrow endpoint y
   coordinate is `apply(sw, line_index × 128) / 128`. Under SW128 the
   rainbow reshuffles; under NO_SWIZZLE the arrows are horizontal.
3. **SMEM physical lines** — stack ordered by physical line index. Each
   line is colored by its *source* GMEM row, so the reader sees the
   shuffle visually.

Below the three tracks: an **element-level bank legend** showing one GMEM
row (32 × 4-byte words for a 128 B coalesced load) colored by target SMEM
bank after the swizzle remap. This is the visual answer to "which word goes
in which bank under swizzle?" — exactly the question TMA/cp.async.bulk.tensor
has to answer at runtime when writing into SMEM.

The C-tile destination track at the bottom mirrors the same idea for the
store path: during `epilogue.tma.store`, a green drain bar shows this CTA's
C tile being written to its `(m, n)` location in the problem-sized C tensor.

Related CUTLASS / PTX references:
- `cute/arch/mma_sm90_gmma.hpp` — canonical `Layout_K_SW128_Atom<T>` atoms
  the swizzle map mirrors.
- PTX ISA §9.7.9 `cp.async.bulk.tensor` — TMA descriptor box-dim walking.
- `cute::TMA_LOAD_IM2COL` — not modelled here; we stay with the plain
  box-tile case.

## 12. Simulation model (v4 — what `src/simulation.ts` tracks)

Every panel's animations are driven by a single `WorldState` snapshot returned
by `simulate(SimInput).worldAt(tick)`. The simulator is a pure, event-driven
function: phase boundaries emit effects, and `worldAt` applies all effects
≤ tick plus linear interpolation within the active phase of each stream.

### WorldState fields

- `active`: the current producer / consumer / epilogue phase per stream (null if idle on that stream).
- `progress`: per-stream fraction 0..1 inside the active phase.
- `ring[stage]`: per-stage `{slice, role ∈ {empty, fill, hold, consume}, fillFrac}`. Length 0 for wmma, length 1 for sm_80 mma, length kStages for warpspec.
- `mbar[stage]`: full/empty + lastArriveTick/lastWaitTick. Length matches ring.
- `producerTransfer`: `{kind: tma/cpasync/wmma-load/ldmatrixA/tcgen05-cp/metadata/scale, kSlab, operand, stage, linesLoaded, linesTotal}`. Null outside producer phases.
- `consumerAtom`: `{kSlab, kStep, kAtomInSlab, stage, atomM, atomN, atomFlatIdx, laneWave, maxWays}`. Null outside consumer phases.
- `cTile.accumulated[m][n]`: count of k-steps that have accumulated into that MN atom's C cell. Caps at slabCount × atomsPerStage_K.
- `cTile.epilogueStaged[m][n]` and `epilogueDrained[m][n]`: 0..1 row-major sweeps during `stg_smem` and `tma.store`.
- `warps[]`: per-warp role `{producer/consumer/epilogue/idle}`. Length 1 for mma/wmma, 4 for warpgroup. Under `.ws`, warp 0 = producer, warps 1..3 = consumer when their respective phases are active.
- `cluster`: non-null for `ctaGroup=2` clusters. `{thisCtaRole, peerActive, sharedLoad}`.
- `auxiliary`: `{metadata: bool, scaleA: bool, scaleB: bool}` flags for sparse / block_scaled.

### Execution semantics by variant

- **wmma-direct (sm_70/72/75)**: producer emits `wmma.load_matrix_sync` per k-atom; consumer `wmma.mma.sync` per k-atom; epilogue `wmma.store_matrix_sync`. No ring, no mbar, no SMEM.
- **cpasync-mma (sm_80/89)**: `cp.async` + `ldmatrix` per k-atom; `mma.sync` per k-atom; `st.global` epilogue. Synthetic 1-slot ring.
- **warpspec-SS (wgmma / tcgen05 aSource=smem)**: 1 `tma.load` per slab fills `ring[stage]`; `atomsPerStage_K` mma.step per slab via descriptor. Ring kStages-deep.
- **warpspec-RS (wgmma aSource=rmem)**: same + `atomsPerStage_K` `ldmatrix-A` sub-phases hoisted before each mma.step.
- **warpspec-TS (tcgen05 aSource=tmem)**: same + 1 `tcgen05.cp` sub-phase per slab (SMEM A → TMEM A).
- **cg2**: 1 multicast `tma.load` per slab feeds two CTAs; `cluster.peerActive` set; epilogue writes disjoint halves.
- **.ws (tcgen05 cg1 warpSpecialized=true)**: warp 0 handles producer; warps 1–3 handle consumer; simultaneous.
- **sparse (.sp)**: extra metadata `tma.load` per slab.
- **block_scaled**: extra SFA + SFB `tma.loads` per slab.

### What remains schematic (from plan §I)

- PTX cycle counts are nominal (TMA=4, MMA=2 ticks etc.), not hardware-measured.
- Mbarrier transactions have zero latency.
- TMA descriptor box-dim walking is atomic per phase.
- Lane-wave bank-conflict replay uses the pattern level; sub-warp scheduling within a collision is not tracked.
- RF bank write-port contention is not modelled.
- Cluster-wide TMEM coherence is shown as mirrored state.
- Problem-level CTA grid walking: this CTA fixed at (0, 0).
- Epilogue drain order within a CTA is row-major; real kernels may permute.
- Fragment tracking per lane is a stub (`world.warps[].fragment` is always `'none'`).
