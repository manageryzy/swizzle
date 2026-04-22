# verify/ — Tier 3 correctness harness

Two independent re-ports of cute's Swizzle + Layout math. The TS test suite
diffs its own output against `expected.json`, which is produced by the C++
port here. If the three disagree, the TS port has drifted.

- `verify_standalone.cpp` — **no dependencies**. Hand-port of cute formulas
  straight from the header comments. Always compiles. Produces `expected.json`.
- `verify.cpp` — **needs CUTLASS + libcudacxx**. Calls `cute::Swizzle` /
  `cute::Layout` directly. The strongest check, but skips if CUDA SDK is
  missing.
- `src/verify_parity.test.ts` (in the TS suite) diffs the TS port against
  `expected.json`. Runs automatically on `npm test`.

## What it tests

1. `Swizzle<B,M,S>::apply(off)` for every atom (NO_SWIZZLE, 32B, 64B, 128B,
   128B_BASE32B) over a fixed set of offsets.
2. `CLayout` thread-0 ownership for `SM80_16x8_Row`, `CLayout_64x8`,
   `CLayout_64x128` — the mapping from (thread, value) to (m, n).

## Building

### verify_standalone (always works)

```sh
cd verify
g++ -std=c++17 -O0 verify_standalone.cpp -o verify_standalone
./verify_standalone > expected.json   # or diff against the checked-in copy
```

### verify (cute, needs CUDA SDK)

Requires CUTLASS + libcudacxx (`cuda/std/*`). Typically provided by a full
CUDA toolkit install. Without it, `verify.cpp` will not compile — this is
expected.

```sh
g++ -std=c++17 -Istubs -I/tmp/cutlass/include -I/usr/local/cuda/include \
    -O0 verify.cpp -o verify
./verify > actual_cute.json
diff expected.json actual_cute.json
```

Use `verify.cpp` to catch drift between the header comments (what
`verify_standalone.cpp` encodes) and cute's real implementation.

## Running the cross-check

```sh
./verify_standalone > actual.json
diff expected.json actual.json   # exit 0 = port matches
```

TS parity happens automatically:

```sh
npm test   # includes src/verify_parity.test.ts
```

## What this does NOT verify

- **Matrix descriptor bitfields**: `make_gmma_desc` / `make_umma_desc` build
  descriptors from SMEM-typed tensors, which require a device context. Our
  `src/descriptor.test.ts` checks the bitfield math directly (golden hex
  values hand-computed from the struct definition in `mma_sm{90,100}_desc.hpp`).
- **Access patterns** (`ldmatrix`, `cp.async`): these are caller-supplied,
  not hardware-determined.
- **Actual silicon behaviour**: bugs in cute itself would slip past this.
  That's Tier 4 (SASS disassembly of a real kernel).

## Troubleshooting

- Build error referencing `__CUDA_ARCH__`: check your `-I` points at a cute
  source tree, not a packaged binary-only release.
- Build error about undefined `cute::half_t` / tensor operand types: some
  mma_traits headers transitively pull in CUDA types. If you hit that, trim
  the includes to `<cute/swizzle.hpp>` and `<cute/layout.hpp>` — you'll lose
  the CLayout dump but keep the Swizzle dump.
