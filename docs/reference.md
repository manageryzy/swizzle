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
